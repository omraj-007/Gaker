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
from torchvision.utils import save_image
from tqdm import tqdm
from generator.Generator import Generator as GeneratorModel
from generator.craftadv import Generator
from data.gaker_dataloader import GakerPairDataset, LoaderConfig, SplitConfig, build_gaker_dataloaders
from utils.gaussian_smoothing import get_gaussian_kernel
from utils.source_model_utils import (
    SourceModelConfig,
    build_source_model,
    normalize_imagenet,
)


@dataclass
class ExportConfig:
    dataset_name: str = "cifar10"
    data_root: str = "./datasets"
    base_split: str = "test"
    image_size: int = 32
    scale_size: int = 32
    num_classes: int = 10

    source_model_name: str = "resnet18"
    source_model_checkpoint: Optional[str] = None
    use_imagenet_pretrained: bool = False

    generator_checkpoint: str = "./checkpoints/gaker/ckpt_19_ResNet18_.pt"
    channel: int = 32
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 1

    seen_ratio: float = 0.7
    split_seed: int = 42
    split_strategy: str = "random"
    max_proto_samples_per_class: Optional[int] = None

    target_mode: str = "unknown"

    batch_size: int = 64
    num_workers: int = 4
    max_samples: Optional[int] = None

    eps: float = 16.0 / 255.0
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True

    save_dir: str = "./exported_adv/attack_export"
    save_delta_visualization: bool = True
    save_delta_tensor: bool = True
    delta_magnify: float = 8.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyImageNetValDataset(Dataset):
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


def _resolve_imagenet_root(data_root: str) -> str:
    direct = Path(data_root)
    nested = direct / "imagenet_data"
    if nested.exists():
        return str(nested)
    return str(direct)


def get_transforms(dataset_name: str, image_size: int, scale_size: int):
    dataset_name = dataset_name.lower()

    if dataset_name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
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


def build_base_datasets(cfg: ExportConfig):
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

        if train_dir.exists():
            train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
        else:
            train_ds = datasets.ImageFolder(root=str(root), transform=train_transform)

        if val_dir.exists():
            test_ds = datasets.ImageFolder(root=str(val_dir), transform=test_transform)
        else:
            test_ds = datasets.ImageFolder(root=str(root), transform=test_transform)

        cfg.num_classes = len(train_ds.classes)
        return train_ds, test_ds

    raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")


class LimitedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_samples: int):
        self.base_dataset = base_dataset
        self.max_samples = min(int(max_samples), len(base_dataset))

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int):
        return self.base_dataset[idx]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def targeted_attack_success(logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == target_labels).float()


@torch.no_grad()
def build_adv_images(
    generator: Generator,
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


@torch.no_grad()
def save_delta_vis(delta: torch.Tensor, save_path: str, magnify: float) -> None:
    delta_vis = torch.clamp(delta * magnify + 0.5, 0.0, 1.0)
    save_image(delta_vis, save_path)

def load_generator_training_info(generator_checkpoint: str) -> Dict:
    ckpt_path = Path(generator_checkpoint)
    ckpt_dir = ckpt_path.parent

    dataset_info_path = ckpt_dir / "dataset_info.json"
    targets_path = ckpt_dir / "targets.json"

    info: Dict = {}

    if dataset_info_path.exists():
        with open(dataset_info_path, "r") as f:
            info.update(json.load(f))

    if targets_path.exists():
        with open(targets_path, "r") as f:
            targets_obj = json.load(f)
        if isinstance(targets_obj, dict) and "targets" in targets_obj:
            info["targets"] = targets_obj["targets"]

    return info


def resolve_export_split_cfg(cfg: ExportConfig, training_info: Dict) -> SplitConfig:
    known_override = None

    if isinstance(training_info.get("targets"), list) and len(training_info["targets"]) > 0:
        known_override = [int(x) for x in training_info["targets"]]

    return SplitConfig(
        seen_ratio=cfg.seen_ratio,
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_strategy,
        known_classes_override=known_override,
        samples_per_known_class=None,
    )

def resolve_export_loader(
    cfg: ExportConfig,
    train_dataset,
    test_dataset,
    feature_extractor,
):

    training_info = load_generator_training_info(cfg.generator_checkpoint)
    split_cfg = resolve_export_split_cfg(cfg, training_info)

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
        feature_extractor=feature_extractor if cfg.split_strategy.lower() == "greedy" else None,
        device=cfg.device,
        max_proto_samples_per_class=cfg.max_proto_samples_per_class,
        topk_targets_per_known_class=None,
        classifier_for_quality_filter=None,
    )

    if cfg.base_split.lower() == "test":
        if cfg.target_mode.lower() == "known":
            pair_dataset = bundle["eval_known_dataset"]
        elif cfg.target_mode.lower() == "unknown":
            pair_dataset = bundle["eval_unknown_dataset"]
        elif cfg.target_mode.lower() == "all":
            pair_dataset = GakerPairDataset(
                base_dataset=test_dataset,
                known_classes=bundle["known_classes"],
                unknown_classes=bundle["unknown_classes"],
                target_mode="all",
                target_pool_by_class=None,
                seed=cfg.split_seed + 3000,
            )
        else:
            raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")

        if cfg.max_samples is not None:
            pair_dataset = LimitedDataset(pair_dataset, cfg.max_samples)

        loader = DataLoader(
            pair_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # return bundle["known_classes"], bundle["unknown_classes"], loader
    return training_info, bundle["known_classes"], bundle["unknown_classes"], loader


def main(cfg: ExportConfig):
    set_seed(cfg.seed)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    if cfg.dataset_name.lower() == "imagenet":
        cfg.use_imagenet_pretrained = True
        cfg.source_model_checkpoint = None
        if cfg.num_classes == 10:
            cfg.num_classes = 1000

    ensure_dir(cfg.save_dir)
    source_dir = os.path.join(cfg.save_dir, "source")
    target_dir = os.path.join(cfg.save_dir, "target")
    adv_dir = os.path.join(cfg.save_dir, "adv")
    delta_vis_dir = os.path.join(cfg.save_dir, "delta_vis")
    delta_tensor_dir = os.path.join(cfg.save_dir, "delta_tensor")

    for p in [source_dir, target_dir, adv_dir]:
        ensure_dir(p)
    if cfg.save_delta_visualization:
        ensure_dir(delta_vis_dir)
    if cfg.save_delta_tensor:
        ensure_dir(delta_tensor_dir)

    train_dataset, test_dataset = build_base_datasets(cfg)

    source_model_cfg = SourceModelConfig(
        model_name=cfg.source_model_name,
        num_classes=cfg.num_classes,
        checkpoint_path=cfg.source_model_checkpoint,
        device=cfg.device,
        use_imagenet_pretrained=cfg.use_imagenet_pretrained,
        freeze=True,
    )
    source_model, source_model_meta = build_source_model(source_model_cfg)

    training_info, known_classes, unknown_classes, export_loader = resolve_export_loader(
        cfg=cfg,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        feature_extractor=source_model,
    )

    if isinstance(training_info.get("targets"), list):
        print(f"Loaded trained known-class split from checkpoint metadata: {len(training_info['targets'])} classes")


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

    kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(cfg.device)
    use_amp = bool(cfg.use_amp and str(cfg.device).startswith("cuda"))

    export_count = 0
    total_success = 0.0
    total_count = 0

    meta_path = os.path.join(cfg.save_dir, "metadata.jsonl")
    summary_path = os.path.join(cfg.save_dir, "export_summary.json")

    with open(meta_path, "w") as meta_file:
        progress = tqdm(export_loader, dynamic_ncols=True, desc="Exporting adversarial dataset")

        for source_images, source_labels, target_images, target_labels in progress:
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
            success = targeted_attack_success(adv_logits, target_labels)
            preds = adv_logits.argmax(dim=1)

            batch_size = source_images.size(0)
            total_success += float(success.sum().item())
            total_count += batch_size

            for i in range(batch_size):
                sample_idx = export_count
                stem = f"sample_{sample_idx:06d}"

                source_rel = f"source/{stem}.png"
                target_rel = f"target/{stem}.png"
                adv_rel = f"adv/{stem}.png"
                delta_vis_rel = f"delta_vis/{stem}.png"
                delta_tensor_rel = f"delta_tensor/{stem}.pt"

                save_image(source_images[i].cpu(), os.path.join(cfg.save_dir, source_rel))
                save_image(target_images[i].cpu(), os.path.join(cfg.save_dir, target_rel))
                save_image(adv_images[i].cpu(), os.path.join(cfg.save_dir, adv_rel))

                if cfg.save_delta_visualization:
                    save_delta_vis(delta[i].cpu(), os.path.join(cfg.save_dir, delta_vis_rel), cfg.delta_magnify)
                else:
                    delta_vis_rel = None

                if cfg.save_delta_tensor:
                    torch.save(delta[i].cpu(), os.path.join(cfg.save_dir, delta_tensor_rel))
                else:
                    delta_tensor_rel = None

                record = {
                    "sample_id": sample_idx,
                    "source_label": int(source_labels[i].item()),
                    "target_label": int(target_labels[i].item()),
                    "adv_pred_label": int(preds[i].item()),
                    "targeted_success": bool(success[i].item() > 0.5),
                    "linf": float(delta[i].abs().max().item()),
                    "l2": float(delta[i].flatten().norm(p=2).item()),
                    "source_image": source_rel,
                    "target_image": target_rel,
                    "adv_image": adv_rel,
                    "delta_visualization": delta_vis_rel,
                    "delta_tensor": delta_tensor_rel,
                }
                meta_file.write(json.dumps(record) + "\n")
                export_count += 1

                if cfg.max_samples is not None and export_count >= cfg.max_samples:
                    break

            progress.set_postfix({
                "saved": export_count,
                "ASR": f"{(total_success / max(1, total_count)) * 100:.2f}%"
            })

            if cfg.max_samples is not None and export_count >= cfg.max_samples:
                break

    summary = {
        "dataset_name": cfg.dataset_name,
        "data_root": cfg.data_root,
        "base_split": cfg.base_split,
        "target_mode": cfg.target_mode,
        "source_model_name": cfg.source_model_name,
        "source_model_checkpoint": cfg.source_model_checkpoint,
        "use_imagenet_pretrained": cfg.use_imagenet_pretrained,
        "generator_checkpoint": cfg.generator_checkpoint,
        "channel": cfg.channel,
        "channel_mult": list(cfg.channel_mult),
        "num_res_blocks": cfg.num_res_blocks,
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "eps": cfg.eps,
        "num_exported_samples": export_count,
        "targeted_asr_over_exported_set": (total_success / total_count) if total_count > 0 else 0.0,
        "metadata_file": meta_path,
        "source_model_metadata_keys": list(source_model_meta.keys()) if isinstance(source_model_meta, dict) else [],
                "used_training_targets_override": bool(isinstance(training_info.get("targets"), list) and len(training_info["targets"]) > 0),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print("Exported adversarial dataset")
    print(f"Target mode: {cfg.target_mode}")
    print(f"Known classes:   {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Exported samples: {export_count}")
    print(f"Targeted ASR over exported set: {summary['targeted_asr_over_exported_set'] * 100:.2f}%")
    print(f"Metadata: {meta_path}")
    print(f"Summary:  {summary_path}")
    print("=" * 100)


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export per-sample adversarial images from the rebuilt attack pipeline")

    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--base_split", type=str, default="test", choices=["train", "test"])
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

    parser.add_argument("--target_mode", type=str, default="unknown", choices=["known", "unknown", "all"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--eps", type=float, default=16.0 / 255.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--save_dir", type=str, default="./exported_adv/attack_export")
    parser.add_argument("--save_delta_visualization", action="store_true")
    parser.add_argument("--save_delta_tensor", action="store_true")
    parser.add_argument("--delta_magnify", type=float, default=8.0)

    args = parser.parse_args()

    return ExportConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        base_split=args.base_split,
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
        target_mode=args.target_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        eps=args.eps,
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
        save_delta_visualization=args.save_delta_visualization,
        save_delta_tensor=args.save_delta_tensor,
        delta_magnify=args.delta_magnify,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
