"""Train CluMuT on the Hugging Face CIFAR-10 dataset.

The dataset is cached under ``--data-root`` on first run.
Use ``--max-epochs 1 --limit-train-batches 5`` for a short smoke run.
"""

import argparse
import re
from functools import partial
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

import stable_pretraining as spt


class TwoViewTransform:
    """Apply the same augmentation pipeline independently to two views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = sample["image"]
        return {
            "views": (self.transform(image), self.transform(image)),
            "label": sample["label"],
        }


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


def linear_evaluate(
    encoder, train_loader, test_loader, device, epochs, log_every, pretrain_epoch, seed
):
    """Train a linear classifier on frozen features and periodically report test accuracy."""
    if log_every < 1:
        raise ValueError("linear_eval_log_every must be positive")
    torch.manual_seed(seed)
    encoder.requires_grad_(False)
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        classifier.train()
        permutation = torch.randperm(train_features.shape[0])
        for indices in permutation.split(train_loader.batch_size):
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(classifier(features), labels).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            classifier.eval()
            correct = 0
            with torch.no_grad():
                for features, labels in zip(
                    test_features.split(test_loader.batch_size),
                    test_labels.split(test_loader.batch_size),
                ):
                    predictions = classifier(features.to(device)).argmax(dim=1).cpu()
                    correct += (predictions == labels).sum().item()
            accuracy = correct / len(test_labels)
            print(
                f"Pretraining epoch {pretrain_epoch} | "
                f"linear eval epoch {epoch + 1}/{epochs}: "
                f"CIFAR-10 test top-1 accuracy: {accuracy:.4%}"
            )


def main():
    parser = argparse.ArgumentParser(description="Train CluMuT on CIFAR-10.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--linear-eval-epochs", type=int, default=100)
    parser.add_argument("--linear-eval-log-every", type=int, default=10)
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
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.encoder_checkpoint and args.linear_eval_epochs < 1:
        parser.error("--linear-eval-epochs must be positive with --encoder-checkpoint")
    pl.seed_everything(args.seed, workers=True)

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
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
            transform=TwoViewTransform(transform),
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
        optim = {
            "optimizer": partial(
                spt.optim.LARS,
                lr=0.5,
                momentum=0.9,
                weight_decay=1e-4,
            ),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0005,
            },
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
        training_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clumut-epoch{epoch:03d}",
            every_n_epochs=args.checkpoint_every,
            save_top_k=0,
            save_last=True,
        )
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            limit_train_batches=(
                args.limit_train_batches
                if args.limit_train_batches is not None
                else 1.0
            ),
            accelerator=args.accelerator,
            callbacks=[checkpoint_callback, training_checkpoint_callback],
        )
        trainer.fit(
            model,
            train_dataloaders=loader,
            ckpt_path=args.resume_from_checkpoint,
        )
        evaluation_checkpoints = checkpoint_callback.saved_checkpoints

    if args.linear_eval_epochs > 0:
        linear_transform = SingleViewTransform(evaluation_transform)
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
            model.backbone.load_state_dict(encoder_state)
            linear_evaluate(
                model.backbone,
                train_eval_loader,
                test_eval_loader,
                device,
                args.linear_eval_epochs,
                args.linear_eval_log_every,
                pretrain_epoch,
                args.seed,
            )


if __name__ == "__main__":
    main()
