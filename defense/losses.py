from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_pix: float = 1.0
    lambda_src_feat: float = 1.0
    lambda_cls: float = 0.25
    lambda_anti_tgt: float = 0.0


@torch.no_grad()
def target_hit_rate(logits: torch.Tensor, target_labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == target_labels).float().mean().item())


@torch.no_grad()
def label_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def feature_cosine_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    feat_a = F.normalize(feat_a, dim=1)
    feat_b = F.normalize(feat_b, dim=1)
    return (1.0 - (feat_a * feat_b).sum(dim=1)).mean()


def anti_target_cosine_loss(feat_hat: torch.Tensor, feat_tgt: torch.Tensor) -> torch.Tensor:
    feat_hat = F.normalize(feat_hat, dim=1)
    feat_tgt = F.normalize(feat_tgt, dim=1)
    cosine_sim = (feat_hat * feat_tgt).sum(dim=1)
    return cosine_sim.mean()


def compute_defense_losses(
    x_hat: torch.Tensor,
    x_source: torch.Tensor,
    source_logits: torch.Tensor,
    purified_logits: torch.Tensor,
    source_features: torch.Tensor,
    purified_features: torch.Tensor,
    source_labels: torch.Tensor,
    weights: LossWeights,
    target_features: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    loss_pix = F.l1_loss(x_hat, x_source)
    loss_src_feat = feature_cosine_loss(purified_features, source_features)
    loss_cls = F.cross_entropy(purified_logits, source_labels)

    total = (
        weights.lambda_pix * loss_pix
        + weights.lambda_src_feat * loss_src_feat
        + weights.lambda_cls * loss_cls
    )

    loss_dict: Dict[str, torch.Tensor] = {
        "loss_total": total,
        "loss_pix": loss_pix,
        "loss_src_feat": loss_src_feat,
        "loss_cls": loss_cls,
    }

    if target_features is not None and float(weights.lambda_anti_tgt) > 0.0:
        loss_anti_tgt = anti_target_cosine_loss(purified_features, target_features)
        total = total + weights.lambda_anti_tgt * loss_anti_tgt
        loss_dict["loss_total"] = total
        loss_dict["loss_anti_tgt"] = loss_anti_tgt

    return loss_dict
