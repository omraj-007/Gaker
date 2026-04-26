from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class SourceModelConfig:
    model_name: str = "resnet18"          # resnet18 / resnet50 / densenet121
    num_classes: int = 10
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    use_imagenet_pretrained: bool = False
    freeze: bool = True


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize RGB images using ImageNet mean/std.

    Input:
        x: tensor of shape [B, 3, H, W] in [0, 1]
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = x.clone()
    x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
    x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
    x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
    return x


class SourceModelBase(nn.Module):
    """
    Common interface used by the attack pipeline.

    forward_features(x):
        returns penultimate feature vectors

    forward_logits(x):
        returns classifier logits
    """

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(x)


class ResNet18Wrapper(SourceModelBase):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.backbone.fc(feats)

    @property
    def feature_dim(self) -> int:
        return int(self.backbone.fc.in_features)


class ResNet50Wrapper(SourceModelBase):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.backbone.fc(feats)

    @property
    def feature_dim(self) -> int:
        return int(self.backbone.fc.in_features)


class DenseNet121Wrapper(SourceModelBase):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.backbone.classifier(feats)

    @property
    def feature_dim(self) -> int:
        return int(self.backbone.classifier.in_features)


def _safe_load_resnet18(use_imagenet_pretrained: bool) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=use_imagenet_pretrained)


def _safe_load_resnet50(use_imagenet_pretrained: bool) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
        return models.resnet50(weights=weights)
    except AttributeError:
        return models.resnet50(pretrained=use_imagenet_pretrained)


def _safe_load_densenet121(use_imagenet_pretrained: bool) -> nn.Module:
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
        return models.densenet121(weights=weights)
    except AttributeError:
        return models.densenet121(pretrained=use_imagenet_pretrained)


def _should_replace_classifier(use_imagenet_pretrained: bool, num_classes: int, checkpoint_path: Optional[str]) -> bool:
    """
    Preserve the original pretrained ImageNet classifier head when we are using
    torchvision ImageNet weights directly for evaluation.

    Replace the head when:
      - we are not using torchvision pretrained weights, or
      - a custom classifier checkpoint will be loaded, or
      - the requested class count is not ImageNet-1K.
    """
    if checkpoint_path is not None and checkpoint_path != "":
        return True
    if not use_imagenet_pretrained:
        return True
    return int(num_classes) != 1000

def build_backbone(model_name: str, num_classes: int, use_imagenet_pretrained: bool = False) -> nn.Module:
    name = model_name.lower()

    if name == "resnet18":
        backbone = _safe_load_resnet18(use_imagenet_pretrained)
        if not (use_imagenet_pretrained and num_classes == 1000):
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone

    if name == "resnet50":
        backbone = _safe_load_resnet50(use_imagenet_pretrained)
        if not (use_imagenet_pretrained and num_classes == 1000):
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone

    if name == "densenet121":
        backbone = _safe_load_densenet121(use_imagenet_pretrained)
        if not (use_imagenet_pretrained and num_classes == 1000):
            backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        return backbone

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Use resnet18, resnet50, or densenet121."
    )

def wrap_source_model(backbone: nn.Module, model_name: str) -> SourceModelBase:
    name = model_name.lower()

    if name == "resnet18":
        return ResNet18Wrapper(backbone)

    if name == "resnet50":
        return ResNet50Wrapper(backbone)

    if name == "densenet121":
        return DenseNet121Wrapper(backbone)

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Use resnet18, resnet50, or densenet121."
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu", strict: bool = False) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            maybe_tensor_values = all(torch.is_tensor(v) for v in checkpoint.values()) if len(checkpoint) > 0 else False
            if maybe_tensor_values:
                state_dict = checkpoint
    elif hasattr(checkpoint, "keys"):
        maybe_tensor_values = all(torch.is_tensor(v) for v in checkpoint.values()) if len(checkpoint) > 0 else False
        if maybe_tensor_values:
            state_dict = checkpoint

    if state_dict is None:
        raise ValueError(
            "Could not find model weights in checkpoint. "
            "Expected keys like 'model_state_dict' or 'state_dict'."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(missing) > 0:
        print(f"[Warning] Missing keys while loading checkpoint: {missing}")
    if len(unexpected) > 0:
        print(f"[Warning] Unexpected keys while loading checkpoint: {unexpected}")

    return checkpoint if isinstance(checkpoint, dict) else {"raw_checkpoint": checkpoint}


def build_source_model(cfg: SourceModelConfig) -> Tuple[SourceModelBase, Dict[str, Any]]:
    """
    Build a classifier wrapper that exposes:
      - forward_features(x)
      - forward_logits(x)

    Usage pattern:
      - ImageNet: set use_imagenet_pretrained=True and checkpoint_path=None
      - CIFAR/Tiny-ImageNet: set use_imagenet_pretrained=False and checkpoint_path=...
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"

    backbone = build_backbone(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        use_imagenet_pretrained=cfg.use_imagenet_pretrained,
        # checkpoint_path=cfg.checkpoint_path,
    )

    metadata: Dict[str, Any] = {}
    if cfg.checkpoint_path is not None and cfg.checkpoint_path != "":
        metadata = load_checkpoint(
            model=backbone,
            checkpoint_path=cfg.checkpoint_path,
            device=device,
            strict=False,
        )

    wrapped = wrap_source_model(backbone, cfg.model_name).to(device)

    if cfg.freeze:
        freeze_model(wrapped)

    return wrapped, metadata


@torch.no_grad()
def infer_feature_dim(model: SourceModelBase, image_size: int, device: str = "cpu") -> int:
    x = torch.randn(2, 3, image_size, image_size, device=device)
    feats = model.forward_features(x)
    return int(feats.shape[1])


@torch.no_grad()
def sanity_check_source_model(
    model: SourceModelBase,
    num_classes: int,
    image_size: int,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    logits = model.forward_logits(x)
    feats = model.forward_features(x)
    return {
        "input_shape": tuple(x.shape),
        "logits_shape": tuple(logits.shape),
        "features_shape": tuple(feats.shape),
        "expected_num_classes": int(num_classes),
        "feature_dim": int(feats.shape[1]),
    }
