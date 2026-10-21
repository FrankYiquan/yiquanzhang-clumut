"""Train CluMuT on the Hugging Face CIFAR-10 dataset.

The dataset is cached under ``--data-root`` on first run.
Use ``--max-epochs 1 --limit-train-batches 5`` for a short smoke run.
"""

import argparse
import math
import random
import re
from functools import partial
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms

import stable_pretraining as spt


class TwoViewTransform:
    """Apply two augmentation pipelines to separate views of one image."""

    def __init__(self, transform, transform_prime):
        self.transform = transform
        self.transform_prime = transform_prime

    def __call__(self, sample):
        image = sample["image"]
        return {
            "views": (self.transform(image), self.transform_prime(image)),
            "label": sample["label"],
        }


class GaussianBlur:
    """Apply Barlow Twins-style Gaussian blur to a PIL image probabilistically."""

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(random.random() * 1.9 + 0.1))
        return image


class Solarization:
    """Apply Barlow Twins-style PIL solarization probabilistically."""

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return ImageOps.solarize(image)
        return image


class BarlowLARS(torch.optim.Optimizer):
    """LARS optimizer and weight/bias parameter groups from Barlow Twins."""

    def __init__(
        self,
        params,
        base_lr: float,
        learning_rate_weights: float = 0.2,
        learning_rate_biases: float = 0.0048,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        eta: float = 0.001,
    ):
        parameters = list(params)
        weight_parameters = [
            parameter for parameter in parameters if parameter.ndim != 1
        ]
        bias_parameters = [parameter for parameter in parameters if parameter.ndim == 1]
        groups = [
            {
                "params": weight_parameters,
                "lr": base_lr * learning_rate_weights,
                "lr_multiplier": learning_rate_weights,
            },
            {
                "params": bias_parameters,
                "lr": base_lr * learning_rate_biases,
                "lr_multiplier": learning_rate_biases,
            },
        ]
        defaults = dict(
            lr=0.0,
            base_lr=base_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=True,
            lars_adaptation_filter=True,
        )
        super().__init__(groups, defaults)

    @staticmethod
    def exclude_bias_and_norm(parameter):
        return parameter.ndim == 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue

                if not group["weight_decay_filter"] or not self.exclude_bias_and_norm(
                    parameter
                ):
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])

                if not group[
                    "lars_adaptation_filter"
                ] or not self.exclude_bias_and_norm(parameter):
                    parameter_norm = torch.norm(parameter)
                    update_norm = torch.norm(gradient)
                    one = torch.ones_like(parameter_norm)
                    adaptation = torch.where(
                        parameter_norm > 0,
                        torch.where(
                            update_norm > 0,
                            group["eta"] * parameter_norm / update_norm,
                            one,
                        ),
                        one,
                    )
                    gradient = gradient.mul(adaptation)

                state = self.state[parameter]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(parameter)
                momentum = state["mu"]
                momentum.mul_(group["momentum"]).add_(gradient)
                parameter.add_(momentum, alpha=-group["lr"])

        return loss


class BarlowWarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """Batch-scaled Barlow Twins warmup and cosine learning-rate schedule."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, base_lr: float):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = min(
                1.0,
                (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps),
            )
            scale = 0.001 + 0.999 * (1 + math.cos(math.pi * progress)) / 2
        return [
            self.base_lr * group["lr_multiplier"] * scale
            for group in self.optimizer.param_groups
        ]


class SingleViewTransform:
    """Return one deterministic normalized image for linear evaluation."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return {"image": self.transform(sample["image"]), "label": sample["label"]}


class EncoderCheckpointCallback(pl.Callback):
    """Save encoder-only snapshots at fixed pretraining epoch intervals."""

    def __init__(self, directory: Path, interval: int):
        self.directory = directory
        self.interval = interval
        self.saved_checkpoints = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0 and epoch != trainer.max_epochs:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"encoder_epoch_{epoch:03d}.pt"
        state = {
            name: value.detach().cpu()
            for name, value in pl_module.backbone.state_dict().items()
        }
        torch.save(state, path)
        self.saved_checkpoints.append((epoch, path))
        print(f"Saved encoder checkpoint: {path}")


class TrainingCheckpointCallback(pl.Callback):
    """Save one rolling full checkpoint for resuming training."""

    def __init__(self, directory: Path, interval: int):
        self.directory = directory
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0 and epoch != trainer.max_epochs:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "last.ckpt"
        trainer.save_checkpoint(str(path))
        print(f"Saved resumable training checkpoint: {path}")


class CluMuTDiagnosticsCallback(pl.Callback):
    """Report CluMuT loss, assignment, correlation, and gradient diagnostics."""

    def __init__(self, log_every: int = 100, initial_batches: int = 3):
        self.log_every = log_every
        self.initial_batches = initial_batches
        self.gradient_squared_norms = {}
        self.hooks = []

    def on_fit_start(self, trainer, pl_module):
        def capture_gradient(name):
            def hook(gradient):
                squared_norm = gradient.detach().float().square().sum()
                self.gradient_squared_norms[name] = (
                    self.gradient_squared_norms.get(name, 0.0) + squared_norm
                )

            return hook

        for name, parameter in pl_module.projector.named_parameters():
            self.hooks.append(parameter.register_hook(capture_gradient("projector")))
        self.hooks.append(
            pl_module.prototypes.register_hook(capture_gradient("prototypes"))
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.gradient_squared_norms.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return

        epoch = trainer.current_epoch + 1
        step = trainer.global_step
        mutual = outputs["mutual_targeting_loss"]
        identitization = outputs["identitization_loss"]
        total = outputs["loss"]
        h_a, h_b = outputs["h_a"], outputs["h_b"]
        q_a, q_b = outputs["q_a"], outputs["q_b"]
        correlation = outputs["correlation"]

        finite_losses = torch.isfinite(torch.stack((mutual, identitization, total)))
        if not finite_losses.all():
            print(f"WARNING epoch={epoch} step={step}: non-finite CluMuT loss")
        if not torch.isfinite(q_a).all() or not torch.isfinite(q_b).all():
            print(f"WARNING epoch={epoch} step={step}: non-finite assignments")
        if correlation is not None and not torch.isfinite(correlation).all():
            print(f"WARNING epoch={epoch} step={step}: non-finite correlation matrix")

        entropy_a = -(q_a * q_a.clamp_min(torch.finfo(q_a.dtype).tiny).log()).sum(-1)
        entropy_b = -(q_b * q_b.clamp_min(torch.finfo(q_b.dtype).tiny).log()).sum(-1)
        concentration_a = (q_a.max(dim=-1).values >= 0.99).float().mean()
        concentration_b = (q_b.max(dim=-1).values >= 0.99).float().mean()
        if min(entropy_a.mean().item(), entropy_b.mean().item()) < 0.1:
            print(f"WARNING epoch={epoch} step={step}: assignment entropy is very low")
        if max(concentration_a.item(), concentration_b.item()) >= 0.9:
            print(
                f"WARNING epoch={epoch} step={step}: most samples assign >=99% "
                "probability to one prototype"
            )

        should_log = step <= self.initial_batches or step % self.log_every == 0
        if not should_log:
            return

        with torch.no_grad():
            diag = correlation.diagonal()
            off_mask = ~torch.eye(
                correlation.shape[0], device=correlation.device, dtype=torch.bool
            )
            h_a_std = h_a.float().std().item()
            h_b_std = h_b.float().std().item()
            h_a_prototype_std = h_a.float().std(dim=0).mean().item()
            h_b_prototype_std = h_b.float().std(dim=0).mean().item()
            diagonal_mean = diag.mean().item()
            diagonal_min = diag.min().item()
            diagonal_max = diag.max().item()
            off_diagonal = correlation[off_mask]
            off_diagonal_mean = off_diagonal.abs().mean().item()
            off_diagonal_min = off_diagonal.min().item()
            off_diagonal_max = off_diagonal.max().item()
        gradient_norms = {
            name: squared_norm.sqrt().item()
            for name, squared_norm in self.gradient_squared_norms.items()
        }
        print(
            f"CluMuT diagnostic epoch={epoch} step={step} "
            f"batch={q_a.shape[0]} mutual={mutual.item():.6g} "
            f"identitization={identitization.item():.6g} total={total.item():.6g} "
            f"h_a={tuple(h_a.shape)} h_b={tuple(h_b.shape)} "
            f"q_a={tuple(q_a.shape)} q_b={tuple(q_b.shape)} "
            f"h_a_std={h_a_std:.6f} h_b_std={h_b_std:.6f} "
            f"h_a_mean_prototype_std={h_a_prototype_std:.6f} "
            f"h_b_mean_prototype_std={h_b_prototype_std:.6f} "
            f"q_a_range=({q_a.min().item():.4g},{q_a.max().item():.4g}) "
            f"q_b_range=({q_b.min().item():.4g},{q_b.max().item():.4g}) "
            f"entropy=({entropy_a.mean().item():.4g},{entropy_b.mean().item():.4g}) "
            f"corr_diag_mean={diagonal_mean:.6f} "
            f"corr_diag_minmax={diagonal_min:.6f}/{diagonal_max:.6f} "
            f"corr_offdiag_abs={off_diagonal_mean:.4g} "
            f"corr_offdiag_minmax={off_diagonal_min:.6f}/{off_diagonal_max:.6f} "
            f"grad_norms={gradient_norms}"
        )

    def on_fit_end(self, trainer, pl_module):
        for hook in self.hooks:
            hook.remove()


def extract_features(encoder, loader, device):
    """Extract frozen encoder features and labels to CPU tensors."""
    features = []
    labels = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features.append(encoder(images).cpu())
            labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(labels)


def _validate_linear_eval_data(split, features, labels):
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"{split} features/labels must have shapes [N, D] and [N], "
            f"got {tuple(features.shape)} and {tuple(labels.shape)}"
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split} feature/label count mismatch: "
            f"{features.shape[0]} features, {labels.shape[0]} labels"
        )
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if labels.dtype not in integer_dtypes:
        raise ValueError(
            f"{split} labels must be integer class indices, got {labels.dtype}"
        )
    if labels.numel() and (labels.min() < 0 or labels.max() > 9):
        raise ValueError(f"{split} labels must be in [0, 9]")
    if not torch.isfinite(features).all():
        raise ValueError(f"{split} encoder features contain non-finite values")


def _log_linear_eval_feature_diagnostics(split, features, labels):
    """Log feature scale, variance, norms, cosine similarity, and label counts."""
    with torch.no_grad():
        values = features.detach().float()
        norms = torch.linalg.vector_norm(values, dim=1)
        sample = values[: min(values.shape[0], 256)]
        normalized_sample = F.normalize(sample, dim=1)
        cosine = normalized_sample @ normalized_sample.T
        off_diagonal_mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
        off_diagonal_cosine = cosine[off_diagonal_mask]
        class_counts = torch.bincount(labels.long(), minlength=10).tolist()
        if off_diagonal_cosine.numel():
            cosine_summary = (
                f"{off_diagonal_cosine.mean().item():.6g}/"
                f"{off_diagonal_cosine.std(unbiased=False).item():.6g}/"
                f"{off_diagonal_cosine.min().item():.6g}/"
                f"{off_diagonal_cosine.max().item():.6g}"
            )
        else:
            cosine_summary = "n/a"
        print(
            f"Linear eval feature diagnostic split={split} "
            f"shape={tuple(features.shape)} dtype={features.dtype} "
            f"mean={values.mean().item():.6g} std={values.std(unbiased=False).item():.6g} "
            f"min={values.min().item():.6g} max={values.max().item():.6g} "
            f"max_abs={values.abs().max().item():.6g} "
            f"finite={torch.isfinite(values).all().item()} "
            f"mean_per_feature_std={values.std(dim=0, unbiased=False).mean().item():.6g} "
            f"feature_norm_mean/min/max="
            f"{norms.mean().item():.6g}/{norms.min().item():.6g}/"
            f"{norms.max().item():.6g} "
            f"sample_pairwise_cosine_mean/std/min/max="
            f"{cosine_summary} "
            f"count={features.shape[0]} label_count={labels.shape[0]} "
            f"label_dtype={labels.dtype} label_min/max="
            f"{labels.min().item()}/{labels.max().item()} "
            f"class_counts={class_counts}"
        )


def _log_linear_eval_classifier_diagnostics(stage, logits, loss, classifier):
    """Log detached classifier output and parameter statistics."""
    with torch.no_grad():
        values = logits.detach().float()
        weight_norm = classifier.weight.detach().float().norm().item()
        bias_norm = classifier.bias.detach().float().norm().item()
        print(
            f"Linear eval classifier diagnostic stage={stage} "
            f"logits_shape={tuple(values.shape)} dtype={logits.dtype} "
            f"logits_mean={values.mean().item():.6g} "
            f"logits_std={values.std(unbiased=False).item():.6g} "
            f"logits_min={values.min().item():.6g} "
            f"logits_max={values.max().item():.6g} "
            f"logits_max_abs={values.abs().max().item():.6g} "
            f"logits_finite={torch.isfinite(values).all().item()} "
            f"cross_entropy={loss.detach().item():.6g} "
            f"weight_norm={weight_norm:.6g} bias_norm={bias_norm:.6g}"
        )


def linear_evaluate(
    encoder,
    train_loader,
    test_loader,
    device,
    epochs,
    log_every,
    pretrain_epoch,
    seed,
    classifier_lr=0.1,
    diagnostics=False,
):
    """Train a linear classifier on frozen features and periodically report metrics."""
    if log_every < 1:
        raise ValueError("linear_eval_log_every must be positive")
    torch.manual_seed(seed)
    encoder.requires_grad_(False)
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    _validate_linear_eval_data("train", train_features, train_labels)
    _validate_linear_eval_data("test", test_features, test_labels)
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            "Train and test encoder feature dimensions differ: "
            f"{train_features.shape[1]} vs {test_features.shape[1]}"
        )
    if diagnostics:
        print(
            "Linear eval data pipeline: train/test share the deterministic "
            "ToTensor+Normalize transform; features and labels are appended "
            "together in non-shuffled loader order."
        )
        _log_linear_eval_feature_diagnostics("train", train_features, train_labels)
        _log_linear_eval_feature_diagnostics("test", test_features, test_labels)
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=classifier_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0
    best_epoch = 0
    final_accuracy = None

    global_step = 0
    total_train_steps = epochs * math.ceil(
        train_features.shape[0] / train_loader.batch_size
    )
    final_gradient_norms = None
    for epoch in range(epochs):
        classifier.train()
        permutation = torch.randperm(train_features.shape[0])
        for batch_index, indices in enumerate(
            permutation.split(train_loader.batch_size)
        ):
            global_step += 1
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            if diagnostics and global_step == 1:
                with torch.no_grad():
                    initial_logits = classifier(features)
                    initial_loss = F.cross_entropy(initial_logits, labels)
                _log_linear_eval_classifier_diagnostics(
                    "before first optimizer update",
                    initial_logits,
                    initial_loss,
                    classifier,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(features), labels)
            loss.backward()
            should_log_gradient = (
                global_step <= 3
                or global_step % 100 == 0
                or global_step == total_train_steps
            )
            if diagnostics and should_log_gradient:
                weight_gradient_norm = (
                    classifier.weight.grad.detach().float().norm().item()
                )
                bias_gradient_norm = classifier.bias.grad.detach().float().norm().item()
                final_gradient_norms = {
                    "weight": weight_gradient_norm,
                    "bias": bias_gradient_norm,
                }
                print(
                    f"Linear eval gradient diagnostic epoch={epoch + 1} "
                    f"step={global_step} batch={batch_index} "
                    f"loss={loss.detach().item():.6g} "
                    f"weight_grad_norm={weight_gradient_norm:.6g} "
                    f"bias_grad_norm={bias_gradient_norm:.6g}"
                )
            optimizer.step()
            if diagnostics and global_step == 1:
                with torch.no_grad():
                    updated_logits = classifier(features)
                    updated_loss = F.cross_entropy(updated_logits, labels)
                _log_linear_eval_classifier_diagnostics(
                    "after first optimizer update",
                    updated_logits,
                    updated_loss,
                    classifier,
                )
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            classifier.eval()

            def evaluate_split(features, labels, batch_size):
                correct = 0
                total_loss = 0.0
                max_abs_logit = 0.0
                with torch.no_grad():
                    for feature_batch, label_batch in zip(
                        features.split(batch_size), labels.split(batch_size)
                    ):
                        logits = classifier(feature_batch.to(device))
                        target = label_batch.to(device)
                        loss = F.cross_entropy(logits, target, reduction="sum")
                        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                            raise ValueError(
                                "Linear probe produced non-finite logits/loss"
                            )
                        total_loss += loss.item()
                        if diagnostics:
                            max_abs_logit = max(
                                max_abs_logit, logits.detach().abs().max().item()
                            )
                        predictions = logits.argmax(dim=1).cpu()
                        correct += (predictions == label_batch).sum().item()
                return total_loss / len(labels), correct / len(labels), max_abs_logit

            train_loss, train_accuracy, train_max_abs_logit = evaluate_split(
                train_features, train_labels, train_loader.batch_size
            )
            test_loss, test_accuracy, test_max_abs_logit = evaluate_split(
                test_features, test_labels, test_loader.batch_size
            )
            final_accuracy = test_accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch + 1
            print(
                f"Pretraining epoch {pretrain_epoch} | "
                f"linear eval epoch {epoch + 1}/{epochs} lr={classifier_lr:g}: "
                f"train loss: {train_loss:.6g} | test loss: {test_loss:.6g} | "
                f"CIFAR-10 train top-1 accuracy: {train_accuracy:.4%} | "
                f"test top-1 accuracy: {test_accuracy:.4%} | "
                f"best so far: {best_accuracy:.4%} at epoch {best_epoch}"
            )
            if diagnostics:
                print(
                    f"Linear eval final-state diagnostic epoch={epoch + 1} "
                    f"train_logits_max_abs={train_max_abs_logit:.6g} "
                    f"test_logits_max_abs={test_max_abs_logit:.6g} "
                    f"weight_norm={classifier.weight.detach().float().norm().item():.6g} "
                    f"bias_norm={classifier.bias.detach().float().norm().item():.6g} "
                    f"last_logged_grad_norms={final_gradient_norms}"
                )

    if final_accuracy is not None:
        print(
            f"Pretraining epoch {pretrain_epoch} | linear evaluation complete: "
            f"best test top-1 accuracy {best_accuracy:.4%} at epoch {best_epoch}; "
            f"final test top-1 accuracy {final_accuracy:.4%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train CluMuT on CIFAR-10.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--linear-eval-epochs", type=int, default=100)
    parser.add_argument("--linear-eval-log-every", type=int, default=10)
    parser.add_argument("--linear-eval-lr", type=float, default=0.1)
    parser.add_argument(
        "--linear-eval-diagnostics",
        action="store_true",
        help="Log frozen-feature, classifier-logit, loss, and gradient diagnostics.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-dir", default="./clumut_checkpoints")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume CluMuT training from a Lightning .ckpt checkpoint.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        help="Evaluate a saved encoder without running self-supervised pretraining.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print loss, assignment, correlation, and gradient diagnostics.",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.encoder_checkpoint and args.linear_eval_epochs < 1:
        parser.error("--linear-eval-epochs must be positive with --encoder-checkpoint")
    if args.linear_eval_lr <= 0:
        parser.error("--linear-eval-lr must be positive")
    pl.seed_everything(args.seed, workers=True)

    shared_augmentations = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    transform = transforms.Compose(
        [
            *shared_augmentations,
            GaussianBlur(probability=1.0),
            Solarization(probability=0.0),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_prime = transforms.Compose(
        [
            *shared_augmentations,
            GaussianBlur(probability=0.1),
            Solarization(probability=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    optim = None
    if not args.encoder_checkpoint:
        dataset = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="train",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=TwoViewTransform(transform, transform_prime),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        steps_per_epoch = len(loader)
        total_steps = max(2, steps_per_epoch * args.max_epochs)
        warmup_steps = min(steps_per_epoch * 10, total_steps - 1)
        base_lr = args.batch_size / 256
        optim = {
            "optimizer": partial(
                BarlowLARS,
                base_lr=base_lr,
                weight_decay=1e-4,
            ),
            "scheduler": partial(
                BarlowWarmupCosineLR,
                warmup_steps=warmup_steps,
                max_steps=total_steps,
                base_lr=base_lr,
            ),
            "interval": "step",
        }

    model = spt.CluMuT(
        encoder_name="resnet18",
        low_resolution=True,
        projector_dims=(1024, 1024),
        n_prototypes=1024,
        temperature=0.1,
        lambd=1.0,
        optim=optim,
    )
    checkpoint_dir = Path(args.checkpoint_dir) / f"batch_size_{args.batch_size}"
    evaluation_checkpoints = []
    if args.encoder_checkpoint:
        checkpoint_path = Path(args.encoder_checkpoint)
        if not checkpoint_path.is_file():
            parser.error(f"encoder checkpoint does not exist: {checkpoint_path}")
        match = re.search(r"encoder_epoch_(\d+)\.pt$", checkpoint_path.name)
        if match is None:
            parser.error(
                "--encoder-checkpoint filename must follow encoder_epoch_NNN.pt"
            )
        evaluation_checkpoints.append((int(match.group(1)), checkpoint_path))
    else:
        checkpoint_callback = EncoderCheckpointCallback(
            checkpoint_dir, args.checkpoint_every
        )
        training_checkpoint_callback = TrainingCheckpointCallback(
            checkpoint_dir, args.checkpoint_every
        )
        callbacks = [checkpoint_callback, training_checkpoint_callback]
        if args.diagnostics:
            callbacks.append(CluMuTDiagnosticsCallback())
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            limit_train_batches=(
                args.limit_train_batches
                if args.limit_train_batches is not None
                else 1.0
            ),
            accelerator=args.accelerator,
            callbacks=callbacks,
        )
        trainer.fit(
            model,
            train_dataloaders=loader,
            ckpt_path=args.resume_from_checkpoint,
        )
        evaluation_checkpoints = checkpoint_callback.saved_checkpoints

    if args.linear_eval_epochs > 0:
        linear_transform = SingleViewTransform(evaluation_transform)
        if args.linear_eval_diagnostics:
            print(
                "Linear-eval dataset validation: train and test use the same "
                "SingleViewTransform and deterministic evaluation transform; "
                "both loaders use shuffle=False."
            )
        linear_train = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="train",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=linear_transform,
        )
        linear_test = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="test",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=linear_transform,
        )
        train_eval_loader = DataLoader(
            linear_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_eval_loader = DataLoader(
            linear_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        use_cuda = args.accelerator in ("gpu", "cuda") or (
            args.accelerator == "auto" and torch.cuda.is_available()
        )
        use_mps = args.accelerator == "mps" and torch.backends.mps.is_available()
        device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
        model.backbone.to(device)
        for pretrain_epoch, checkpoint_path in evaluation_checkpoints:
            encoder_state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.backbone.load_state_dict(encoder_state, strict=True)
            print(
                f"Loaded encoder checkpoint strictly: {checkpoint_path} "
                f"({len(encoder_state)} state entries)"
            )
            linear_evaluate(
                model.backbone,
                train_eval_loader,
                test_eval_loader,
                device,
                args.linear_eval_epochs,
                args.linear_eval_log_every,
                pretrain_epoch,
                args.seed,
                classifier_lr=args.linear_eval_lr,
                diagnostics=args.linear_eval_diagnostics,
            )


if __name__ == "__main__":
    main()
