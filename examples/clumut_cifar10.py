"""Train CluMuT on the Hugging Face CIFAR-10 dataset.

The dataset is cached under ``--data-root`` on first run.
Use ``--max-epochs 1 --limit-train-batches 5`` for a short smoke run.
"""

import argparse
from functools import partial

import lightning as pl
import torch
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


def main():
    parser = argparse.ArgumentParser(description="Train CluMuT on CIFAR-10.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
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
    )

    model = spt.CluMuT(
        encoder_name="resnet18",
        low_resolution=True,
        n_prototypes=1000,
        temperature=0.1,
        optim={
            "optimizer": partial(
                torch.optim.SGD, lr=0.3, momentum=0.9, weight_decay=1e-4
            ),
        },
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches or 1.0,
        accelerator=args.accelerator,
    )
    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
