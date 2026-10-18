"""Focused tests for CluMuT assignments, losses, and training integration."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.losses import CluMuTLoss
from stable_pretraining.methods import CluMuT

pytestmark = pytest.mark.unit


def _model() -> CluMuT:
    backbone = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 8),
    )
    return CluMuT(
        encoder_name=backbone,
        projector_dims=(8, 4),
        n_prototypes=5,
        temperature=0.2,
    )


def test_assignments_and_loss_gradients_reach_encoder_and_prototypes():
    torch.manual_seed(0)
    model = _model()
    view_a = torch.randn(4, 3, 16, 16)
    view_b = torch.randn(4, 3, 16, 16)

    output = model(view_a, view_b)

    assert output.h_a.shape == (4, 5)
    assert output.h_b.shape == (4, 5)
    assert output.q_a.shape == (4, 5)
    assert output.q_b.shape == (4, 5)
    assert torch.allclose(output.q_a.sum(dim=-1), torch.ones(4))
    assert torch.isfinite(output.loss)
    assert torch.allclose(
        output.loss,
        output.mutual_targeting_loss + output.identitization_loss,
    )
    output.loss.backward()

    assert model.prototypes.grad is not None
    assert model.prototypes.grad.abs().sum() > 0
    backbone_weight = model.backbone[-1].weight
    assert backbone_weight.grad is not None
    assert backbone_weight.grad.abs().sum() > 0


def test_identical_views_have_zero_mutual_targeting_loss():
    model = _model().eval()
    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        output = model(images, images)

    assert output.mutual_targeting_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(output.q_a, output.q_b)


def test_different_views_produce_nonzero_mutual_targeting_loss():
    torch.manual_seed(2)
    model = _model().eval()
    view_a = torch.randn(4, 3, 16, 16)
    view_b = torch.randn(4, 3, 16, 16) + 2

    output = model(view_a, view_b)

    assert output.mutual_targeting_loss.item() > 0
    assert not torch.allclose(output.q_a, output.q_b)


def test_identitization_is_finite_for_constant_dimensions_and_rejects_batch_one():
    loss = CluMuTLoss()
    h_a = torch.ones(3, 4, requires_grad=True)
    h_b = torch.ones(3, 4, requires_grad=True)
    result = loss.identitization_loss(h_a, h_b)
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(h_a.grad).all()
    assert torch.isfinite(h_b.grad).all()

    with pytest.raises(ValueError, match="batch size of at least 2"):
        loss.identitization_loss(torch.ones(1, 4), torch.ones(1, 4))


def test_framework_batch_dict_contract_returns_training_state():
    model = _model()
    batch = {"views": [torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16)]}

    state = model(batch, stage="validate")

    assert isinstance(state, dict)
    assert torch.isfinite(state["loss"])
    assert state["embedding"].shape == (8, 8)


class _TwoViewDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(4)
        self.views_a = torch.randn(4, 3, 16, 16, generator=generator)
        self.views_b = torch.randn(4, 3, 16, 16, generator=generator)

    def __len__(self):
        return len(self.views_a)

    def __getitem__(self, index):
        return {"views": (self.views_a[index], self.views_b[index])}


def test_lightning_training_step_runs_with_two_view_loader():
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    model = CluMuT(
        encoder_name=nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)
        ),
        projector_dims=(8, 4),
        n_prototypes=5,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader(_TwoViewDataset(), batch_size=4),
    )
