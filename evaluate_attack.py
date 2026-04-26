from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from generator.Generator import Generator as GeneratorModel
from data.gaker_dataloader import LoaderConfig, SplitConfig, GakerPairDataset, build_gaker_dataloaders
from utils.gaussian_smoothing import get_gaussian_kernel
from utils.source_model_utils import (
    SourceModelConfig,
    build_source_model,
    normalize_imagenet,
)
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class EvalConfig:
    dataset_name: str = "cifar10"             # cifar10 / cifar100 / tinyimagenet / imagenet
    data_root: str = "./datasets"
    image_size: int = 32
    scale_size: int = 32
    num_classes: int = 10

    source_model_name: str = "resnet18"       # resnet18 / resnet50 / densenet121
    source_model_checkpoint: Optional[str] = None
    use_imagenet_pretrained: bool = False

    generator_checkpoint: str = "./checkpoints/gaker/ckpt_19_ResNet18_.pt"
    channel: int = 32
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 1

    seen_ratio: float = 0.7
    split_seed: int = 42
    split_strategy: str = "random"            # random / greedy
    max_proto_samples_per_class: Optional[int] = None
    use_generator_training_split: bool = True

    batch_size: int = 64
    num_workers: int = 4
    max_eval_samples_per_split: Optional[int] = 1000

    eps: float = 16.0 / 255.0
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True

    save_dir: str = "./eval_outputs/attack_eval"
    save_examples: bool = True
    num_example_batches: int = 2
    delta_magnify: float = 8.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyImageNetValDataset(Dataset):
    """
    Supports the official raw Tiny-ImageNet validation layout:

      tiny-imagenet-200/
        train/<wnid>/...
        val/images/*.JPEG
        val/val_annotations.txt
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        images_dir = os.path.join(val_dir, "images")
        ann_path = os.path.join(val_dir, "val_annotations.txt")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_dir}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Tiny-ImageNet val/images directory not found: {images_dir}")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Tiny-ImageNet val_annotations.txt not found: {ann_path}")

        self.classes = sorted(
            [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                wnid = parts[1]
                if wnid not in self.class_to_idx:
                    continue
                img_path = os.path.join(images_dir, img_name)
                label = self.class_to_idx[wnid]
                self.samples.append((img_path, label))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class LimitedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_samples: int):
        self.base_dataset = base_dataset
        self.max_samples = min(int(max_samples), len(base_dataset))

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int):
        return self.base_dataset[idx]


def _resolve_imagenet_root(data_root: str) -> str:
    """
    Accept either:
      - ./datasets/imagenet_data
      - ./datasets  (if imagenet_data is inside)
    """
    direct = Path(data_root)
    nested = direct / "imagenet_data"

    if nested.exists():
        return str(nested)
    return str(direct)


def get_transforms(dataset_name: str, image_size: int, scale_size: int):
    dataset_name = dataset_name.lower()

    if dataset_name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_transform, test_transform

    if dataset_name in {"tinyimagenet", "imagenet"}:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)) if scale_size == image_size else transforms.Resize(scale_size),
            transforms.CenterCrop(image_size) if scale_size != image_size else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)) if scale_size == image_size else transforms.Resize(scale_size),
            transforms.CenterCrop(image_size) if scale_size != image_size else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        return train_transform, test_transform

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


from torch.utils.data import Dataset

def load_imagenet_wnid_to_official_idx(mapping_path: str):
    wnids = []
    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid = line.split()[0]
            wnids.append(wnid)

    if len(wnids) != 1000:
        raise ValueError(f"Expected 1000 wnids in {mapping_path}, got {len(wnids)}")

    return {wnid: idx for idx, wnid in enumerate(wnids)}


def build_imagefolder_to_official_imagenet_remap(imagefolder_dataset, mapping_path: str):
    wnid_to_official = load_imagenet_wnid_to_official_idx(mapping_path)

    remap = {}
    for wnid, imagefolder_idx in imagefolder_dataset.class_to_idx.items():
        if wnid not in wnid_to_official:
            raise KeyError(f"WNID {wnid} not found in {mapping_path}")
        remap[int(imagefolder_idx)] = int(wnid_to_official[wnid])

    return remap


class LabelRemapDataset(Dataset):
    def __init__(self, base_dataset, label_remap):
        self.base_dataset = base_dataset
        self.label_remap = {int(k): int(v) for k, v in label_remap.items()}

        # preserve common dataset attrs if present
        self.classes = getattr(base_dataset, "classes", None)
        self.class_to_idx = getattr(base_dataset, "class_to_idx", None)
        self.samples = getattr(base_dataset, "samples", None)
        self.imgs = getattr(base_dataset, "imgs", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image, int(self.label_remap[int(label)])



def build_base_datasets(cfg: EvalConfig):
    train_transform, test_transform = get_transforms(cfg.dataset_name, cfg.image_size, cfg.scale_size)
    name = cfg.dataset_name.lower()

    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=False, transform=train_transform)
        test_ds = datasets.CIFAR10(root=cfg.data_root, train=False, download=False, transform=test_transform)
        cfg.num_classes = 10
        return train_ds, test_ds

    if name == "cifar100":
        train_ds = datasets.CIFAR100(root=cfg.data_root, train=True, download=False, transform=train_transform)
        test_ds = datasets.CIFAR100(root=cfg.data_root, train=False, download=False, transform=test_transform)
        cfg.num_classes = 100
        return train_ds, test_ds

    if name == "tinyimagenet":
        root = Path(cfg.data_root) / "tiny-imagenet-200"
        train_dir = root / "train"
        val_images_dir = root / "val_images"

        if not train_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet train folder not found: {train_dir}")

        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform)

        if val_images_dir.exists():
            test_ds = datasets.ImageFolder(root=str(val_images_dir), transform=test_transform)
        else:
            test_ds = TinyImageNetValDataset(root=str(root), transform=test_transform)

        cfg.num_classes = len(train_ds.classes)
        return train_ds, test_ds

    if name == "imagenet":
        root = Path(_resolve_imagenet_root(cfg.data_root))

        train_dir = root / "train"
        val_dir = root / "val"

        if train_dir.exists() and val_dir.exists():
            train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
            test_ds = datasets.ImageFolder(root=str(val_dir), transform=test_transform)
        else:
            train_ds = datasets.ImageFolder(root=str(root), transform=train_transform)
            test_ds = datasets.ImageFolder(root=str(root), transform=test_transform)

        cfg.num_classes = len(train_ds.classes)
        return train_ds, test_ds

    raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@torch.no_grad()
def targeted_attack_success_rate(logits: torch.Tensor, target_labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == target_labels).float().mean().item())


@torch.no_grad()
def clean_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def mean_linf(delta: torch.Tensor) -> float:
    return float(delta.abs().flatten(1).max(dim=1).values.mean().item())


@torch.no_grad()
def mean_l2(delta: torch.Tensor) -> float:
    return float(delta.flatten(1).norm(p=2, dim=1).mean().item())


@torch.no_grad()
def build_adv_images(
    generator: GeneratorModel,
    source_model,
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    eps: float,
    kernel,
):
    target_features = source_model.forward_features(normalize_imagenet(target_images))
    perturbated_imgs = generator(source_images, mix=target_features)
    perturbated_imgs = kernel(perturbated_imgs)
    adv = torch.min(torch.max(perturbated_imgs, source_images - eps), source_images + eps)
    adv = torch.clamp(adv, 0.0, 1.0)
    delta = adv - source_images
    return adv, delta


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def save_example_triptych(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    adv_images: torch.Tensor,
    save_path: str,
    max_items: int = 8,
):
    n = min(max_items, source_images.size(0))
    stacked = torch.cat([
        source_images[:n].cpu(),
        target_images[:n].cpu(),
        adv_images[:n].cpu(),
    ], dim=0)
    grid = make_grid(stacked, nrow=n, padding=2)
    save_image(grid, save_path)


@torch.no_grad()
def save_perturbation_grid(
    source_images: torch.Tensor,
    adv_images: torch.Tensor,
    save_path: str,
    max_items: int = 8,
    magnify: float = 8.0,
):
    n = min(max_items, source_images.size(0))
    delta = adv_images[:n] - source_images[:n]
    delta_vis = torch.clamp(delta * magnify + 0.5, 0.0, 1.0)
    grid = make_grid(delta_vis.cpu(), nrow=n, padding=2)
    save_image(grid, save_path)


def _load_generator_training_split(generator_checkpoint: str) -> Dict[str, object]:
    ckpt_path = Path(generator_checkpoint)
    save_dir = ckpt_path.parent

    dataset_info_path = save_dir / "dataset_info.json"
    targets_path = save_dir / "targets.json"

    info: Dict[str, object] = {}
    if dataset_info_path.is_file():
        with dataset_info_path.open("r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            info.update(loaded)

    if targets_path.is_file():
        with targets_path.open("r") as f:
            loaded_targets = json.load(f)
        if isinstance(loaded_targets, dict) and "targets" in loaded_targets:
            info["targets"] = loaded_targets["targets"]

    return info

def _resolve_split_cfg_from_training_metadata(cfg: EvalConfig, training_info: Dict[str, object]) -> SplitConfig:
    known_override = None

    if isinstance(training_info.get("targets"), list):
        # For your setup, saved targets are already the correct class ids.
        known_override = [int(x) for x in training_info["targets"]]

    return SplitConfig(
        seen_ratio=cfg.seen_ratio,
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_strategy,
        known_classes_override=known_override,
        samples_per_known_class=None,
    )

def _maybe_limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None:
        return dataset
    return LimitedDataset(dataset, max_samples=max_samples)

def _build_eval_loader_for_dataset(dataset: Dataset, cfg: EvalConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

def resolve_eval_loaders(
    cfg: EvalConfig,
    train_dataset,
    test_dataset,
    feature_extractor,
):
    training_info = _load_generator_training_split(cfg.generator_checkpoint) if cfg.use_generator_training_split else {}
    split_cfg = _resolve_split_cfg_from_training_metadata(cfg, training_info)

    loader_cfg = LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle_train=False,
        drop_last_train=False,
    )

    bundle = build_gaker_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        split_cfg=split_cfg,
        loader_cfg=loader_cfg,
        feature_extractor=feature_extractor if cfg.split_strategy.lower() == "greedy" and split_cfg.known_classes_override is None else None,
        device=cfg.device,
        max_proto_samples_per_class=cfg.max_proto_samples_per_class,
        topk_targets_per_known_class=None,
        classifier_for_quality_filter=None,
    )

    eval_known_dataset = bundle["eval_known_dataset"]
    eval_unknown_dataset = bundle["eval_unknown_dataset"]

    eval_known_dataset = _maybe_limit_dataset(eval_known_dataset, cfg.max_eval_samples_per_split)
    eval_unknown_dataset = _maybe_limit_dataset(eval_unknown_dataset, cfg.max_eval_samples_per_split)

    eval_known_loader = _build_eval_loader_for_dataset(eval_known_dataset, cfg)
    eval_unknown_loader = _build_eval_loader_for_dataset(eval_unknown_dataset, cfg)

    return training_info, bundle["known_classes"], bundle["unknown_classes"], eval_known_loader, eval_unknown_loader

@torch.no_grad()
def evaluate_loader(
    generator: GeneratorModel,
    source_model,
    loader: DataLoader,
    cfg: EvalConfig,
    split_name: str,
):
    generator.eval()
    if hasattr(source_model, "eval"):
        source_model.eval()

    use_amp = bool(cfg.use_amp and str(cfg.device).startswith("cuda"))

    asr_meter = AverageMeter()
    source_acc_meter = AverageMeter()
    target_acc_meter = AverageMeter()
    linf_meter = AverageMeter()
    l2_meter = AverageMeter()

    saved_batches = 0
    example_dir = os.path.join(cfg.save_dir, split_name)
    if cfg.save_examples:
        ensure_dir(example_dir)

    kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(cfg.device)

    progress = tqdm(loader, dynamic_ncols=True, desc=f"Eval {split_name}")

    for batch_idx, (source_images, source_labels, target_images, target_labels) in enumerate(progress):
        source_images = source_images.to(cfg.device, non_blocking=True)
        source_labels = source_labels.to(cfg.device, non_blocking=True)
        target_images = target_images.to(cfg.device, non_blocking=True)
        target_labels = target_labels.to(cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            adv_images, delta = build_adv_images(
                generator=generator,
                source_model=source_model,
                source_images=source_images,
                target_images=target_images,
                eps=cfg.eps,
                kernel=kernel,
            )

        adv_logits = source_model.forward_logits(normalize_imagenet(adv_images))
        src_logits = source_model.forward_logits(normalize_imagenet(source_images))
        tgt_logits = source_model.forward_logits(normalize_imagenet(target_images))

        batch_size = source_images.size(0)
        asr = targeted_attack_success_rate(adv_logits, target_labels)
        src_acc = clean_accuracy(src_logits, source_labels)
        tgt_acc = clean_accuracy(tgt_logits, target_labels)

        asr_meter.update(asr, batch_size)
        source_acc_meter.update(src_acc, batch_size)
        target_acc_meter.update(tgt_acc, batch_size)
        linf_meter.update(mean_linf(delta), batch_size)
        l2_meter.update(mean_l2(delta), batch_size)

        progress.set_postfix({
            "ASR": f"{asr_meter.avg * 100:.2f}%",
            "src_acc": f"{source_acc_meter.avg * 100:.2f}%",
            "tgt_acc": f"{target_acc_meter.avg * 100:.2f}%",
            "linf": f"{linf_meter.avg:.6f}",
            "l2": f"{l2_meter.avg:.4f}",
        })

        if cfg.save_examples and saved_batches < cfg.num_example_batches:
            save_example_triptych(
                source_images=source_images,
                target_images=target_images,
                adv_images=adv_images,
                save_path=os.path.join(example_dir, f"batch_{batch_idx:03d}_triptych.png"),
                max_items=8,
            )
            save_perturbation_grid(
                source_images=source_images,
                adv_images=adv_images,
                save_path=os.path.join(example_dir, f"batch_{batch_idx:03d}_delta.png"),
                max_items=8,
                magnify=cfg.delta_magnify,
            )
            saved_batches += 1

    return {
        "num_samples": int(asr_meter.count),
        "asr": asr_meter.avg,
        "source_clean_acc": source_acc_meter.avg,
        "target_clean_acc": target_acc_meter.avg,
        "linf": linf_meter.avg,
        "l2": l2_meter.avg,
    }


def main(cfg: EvalConfig):
    set_seed(cfg.seed)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg.save_dir)

    if cfg.dataset_name.lower() == "imagenet":
        cfg.use_imagenet_pretrained = True
        cfg.source_model_checkpoint = None
    if cfg.num_classes == 10:
        cfg.num_classes = 1000
    if cfg.image_size == 32:
        cfg.image_size = 224
    if cfg.scale_size == 32:
        cfg.scale_size = 256

    train_dataset, test_dataset = build_base_datasets(cfg)
    print("Num classes:", cfg.num_classes)
    print("First 10 classes:", getattr(train_dataset, "classes", None)[:10] if hasattr(train_dataset, "classes") else "N/A")

    source_model_cfg = SourceModelConfig(
        model_name=cfg.source_model_name,
        num_classes=cfg.num_classes,
        checkpoint_path=cfg.source_model_checkpoint,
        device=cfg.device,
        use_imagenet_pretrained=cfg.use_imagenet_pretrained,
        freeze=True,
    )
    # source_model, source_model_meta = build_source_model(source_model_cfg)
    if cfg.dataset_name.lower() == "imagenet" and cfg.source_model_name.lower() == "resnet50" and cfg.source_model_checkpoint is None:

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(cfg.device).eval()

        class DirectResNet50Wrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward_features(self, x):
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

            def forward_logits(self, x):
                feats = self.forward_features(x)
                return self.backbone.fc(feats)

            def forward(self, x):
                return self.forward_logits(x)

            @property
            def feature_dim(self):
                return int(self.backbone.fc.in_features)

        source_model = DirectResNet50Wrapper(backbone).to(cfg.device).eval()
        source_model_meta = {}
    else:
        source_model, source_model_meta = build_source_model(source_model_cfg)

    print("use_imagenet_pretrained:", cfg.use_imagenet_pretrained)
    print("source_model_checkpoint:", cfg.source_model_checkpoint)
    print("source model type:", type(source_model).__name__)

    # temporary sanity check: clean accuracy of source model on first 1000 samples
    sanity_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    correct = 0
    total = 0
    for images, labels in sanity_loader:
        images = images.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)

        logits = source_model.forward_logits(normalize_imagenet(images))
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if total >= 1000:
            break

    print("Standalone source-model sanity acc:", correct / total)

    split_cfg = SplitConfig(
        seen_ratio=cfg.seen_ratio,
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_strategy,
    )

    print("use_imagenet_pretrained:", cfg.use_imagenet_pretrained)
    print("source_model_checkpoint:", cfg.source_model_checkpoint)
    print("source model type:", type(source_model).__name__)

    training_info, known_classes, unknown_classes, eval_known_loader, eval_unknown_loader = resolve_eval_loaders(
        cfg=cfg,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        feature_extractor=source_model,
    )

    if not hasattr(source_model, "feature_dim"):
        raise ValueError("source_model wrapper must expose feature_dim property.")
    feature_dim = int(source_model.feature_dim)

    generator = GeneratorModel(
        num_target=len(known_classes),
        ch=cfg.channel,
        ch_mult=list(cfg.channel_mult),
        num_res_blocks=cfg.num_res_blocks,
        feature_channel_num=feature_dim,
    ).to(cfg.device)

    ckpt = torch.load(cfg.generator_checkpoint, map_location=cfg.device)
    generator.load_state_dict(ckpt, strict=False)
    generator.eval()

    known_stats = evaluate_loader(
        generator=generator,
        source_model=source_model,
        loader=eval_known_loader,
        cfg=cfg,
        split_name="known_targets",
    )

    unknown_stats = evaluate_loader(
        generator=generator,
        source_model=source_model,
        loader=eval_unknown_loader,
        cfg=cfg,
        split_name="unknown_targets",
    )

    summary = {
        "dataset_name": cfg.dataset_name,
        "data_root": cfg.data_root,
        "source_model_name": cfg.source_model_name,
        "source_model_checkpoint": cfg.source_model_checkpoint,
        "use_imagenet_pretrained": cfg.use_imagenet_pretrained,
        "generator_checkpoint": cfg.generator_checkpoint,
        "channel": cfg.channel,
        "channel_mult": list(cfg.channel_mult),
        "num_res_blocks": cfg.num_res_blocks,
        "eps": cfg.eps,
        "seen_ratio": cfg.seen_ratio,
        "split_seed": cfg.split_seed,
        "split_strategy": cfg.split_strategy,
        "use_generator_training_split": cfg.use_generator_training_split,
        "max_eval_samples_per_split": cfg.max_eval_samples_per_split,
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "generator_training_info_keys": sorted(list(training_info.keys())),
        "known_targets": known_stats,
        "unknown_targets": unknown_stats,
        "source_model_metadata_keys": list(source_model_meta.keys()) if isinstance(source_model_meta, dict) else [],
    }

    with open(os.path.join(cfg.save_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print("Attack evaluation summary")
    print(f"Known classes:   {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Known samples evaluated:   {known_stats['num_samples']}")
    print(f"Unknown samples evaluated: {unknown_stats['num_samples']}")
    print("-" * 100)
    print(
        f"Known targets   | ASR={known_stats['asr'] * 100:.2f}% | "
        f"source_acc={known_stats['source_clean_acc'] * 100:.2f}% | "
        f"target_acc={known_stats['target_clean_acc'] * 100:.2f}% | "
        f"linf={known_stats['linf']:.6f} | l2={known_stats['l2']:.6f}"
    )
    print(
        f"Unknown targets | ASR={unknown_stats['asr'] * 100:.2f}% | "
        f"source_acc={unknown_stats['source_clean_acc'] * 100:.2f}% | "
        f"target_acc={unknown_stats['target_clean_acc'] * 100:.2f}% | "
        f"linf={unknown_stats['linf']:.6f} | l2={unknown_stats['l2']:.6f}"
    )
    print(f"Saved summary to: {os.path.join(cfg.save_dir, 'evaluation_summary.json')}")
    if cfg.save_examples:
        print(f"Saved example grids under: {cfg.save_dir}")
    print("=" * 100)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate attack on known and unknown target classes")

    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--scale_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--source_model_name", type=str, default="resnet18")
    parser.add_argument("--source_model_checkpoint", type=str, default=None)
    parser.add_argument("--use_imagenet_pretrained", action="store_true")

    parser.add_argument("--generator_checkpoint", type=str, required=True)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_mult", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--num_res_blocks", type=int, default=1)

    parser.add_argument("--seen_ratio", type=float, default=0.7)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_strategy", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--max_proto_samples_per_class", type=int, default=None)
    parser.add_argument("--use_generator_training_split", type=str2bool, default=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_eval_samples_per_split", type=int, default=1000)

    parser.add_argument("--eps", type=float, default=16.0 / 255.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--save_dir", type=str, default="./eval_outputs/attack_eval")
    parser.add_argument("--save_examples", action="store_true")
    parser.add_argument("--num_example_batches", type=int, default=2)
    parser.add_argument("--delta_magnify", type=float, default=8.0)

    args = parser.parse_args()

    return EvalConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        image_size=args.image_size,
        scale_size=args.scale_size,
        num_classes=args.num_classes,
        source_model_name=args.source_model_name,
        source_model_checkpoint=args.source_model_checkpoint,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        generator_checkpoint=args.generator_checkpoint,
        channel=args.channel,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        seen_ratio=args.seen_ratio,
        split_seed=args.split_seed,
        split_strategy=args.split_strategy,
        max_proto_samples_per_class=args.max_proto_samples_per_class,
        use_generator_training_split=args.use_generator_training_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples_per_split=args.max_eval_samples_per_split,
        eps=args.eps,
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
        save_examples=args.save_examples,
        num_example_batches=args.num_example_batches,
        delta_magnify=args.delta_magnify,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
