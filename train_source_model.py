from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm


@dataclass
class TrainConfig:
    dataset_name: str = "cifar10"
    data_root: str = "./datasets"
    save_dir: str = "./checkpoints/source_model"
    model_name: str = "resnet18"
    num_classes: int = 10
    image_size: int = 32
    in_channels: int = 3
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    scheduler: str = "cosine"
    milestones: Tuple[int, ...] = (60, 80)
    gamma: float = 0.1
    label_smoothing: float = 0.0
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True
    save_best_only: bool = True


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
            
        self.classes = sorted([
            d for d in os.listdir(train_dir) 
            if os.path.isdir(os.path.join(train_dir, d))
        ])
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


def get_transforms(dataset_name: str, image_size: int):
    dataset_name = dataset_name.lower()
    
    if dataset_name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_transform, test_transform
        
    if dataset_name == "tinyimagenet":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return train_transform, test_transform
        
    raise ValueError(
        f"Unsupported dataset_name: {dataset_name}. "
        f"Use cifar10, cifar100, or tinyimagenet. "
        f"For imagenet, use pretrained torchvision weights instead."
    )


def build_datasets(cfg: TrainConfig):
    train_transform, test_transform = get_transforms(cfg.dataset_name, cfg.image_size)
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
        train_dir = Path(cfg.data_root) / "tiny-imagenet-200" / "train"
        val_images_dir = Path(cfg.data_root) / "tiny-imagenet-200" / "val_images"
        raw_root = Path(cfg.data_root) / "tiny-imagenet-200"
        
        if not train_dir.exists():
            raise FileNotFoundError(
                "Tiny-ImageNet train folder not found. Expected:\n"
                f" {train_dir}"
            )
            
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
        
        if val_images_dir.exists():
            test_ds = datasets.ImageFolder(root=str(val_images_dir), transform=test_transform)
        else:
            test_ds = TinyImageNetValDataset(root=str(raw_root), transform=test_transform)
            
        cfg.num_classes = len(train_ds.classes)
        return train_ds, test_ds
        
    raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")


def build_loaders(cfg: TrainConfig):
    train_ds, test_ds = build_datasets(cfg)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    
    return train_ds, test_ds, train_loader, test_loader


def adapt_first_conv_if_needed(model: nn.Module, in_channels: int) -> nn.Module:
    if in_channels == 3:
        return model
        
    if not hasattr(model, "conv1"):
        raise ValueError("Model does not expose conv1, so automatic in_channels adaptation is unsupported.")
        
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            repeat = int(np.ceil(in_channels / 3))
            weight = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
            weight = weight * (3.0 / float(in_channels))
            new_conv.weight.copy_(weight)
            
    model.conv1 = new_conv
    return model


def build_model(cfg: TrainConfig) -> nn.Module:
    name = cfg.model_name.lower()
    
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model = adapt_first_conv_if_needed(model, cfg.in_channels)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        return model
        
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model = adapt_first_conv_if_needed(model, cfg.in_channels)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        return model
        
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, cfg.num_classes)
        return model
        
    raise ValueError(
        f"Unsupported model_name: {cfg.model_name}. "
        f"Use resnet18, resnet50, or densenet121."
    )


def build_optimizer(cfg: TrainConfig, model: nn.Module):
    if cfg.optimizer.lower() == "adamw":
        return AdamW(
            model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer.lower() == "sgd":
        return SGD(
            model.parameters(), 
            lr=cfg.lr, 
            momentum=cfg.momentum, 
            weight_decay=cfg.weight_decay, 
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(cfg: TrainConfig, optimizer):
    sched = cfg.scheduler.lower()
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if sched == "multistep":
        return MultiStepLR(optimizer, milestones=list(cfg.milestones), gamma=cfg.gamma)
    if sched == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


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


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer, 
    criterion, 
    device: str, 
    scaler: torch.cuda.amp.GradScaler, 
    use_amp: bool, 
    epoch: int, 
    total_epochs: int,
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    progress = tqdm(loader, dynamic_ncols=True, desc=f"Train {epoch:03d}/{total_epochs:03d}")
    
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = images.size(0)
        batch_acc = accuracy_from_logits(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_acc, batch_size)
        
        progress.set_postfix({
            "loss": f"{loss_meter.avg:.4f}", 
            "acc": f"{acc_meter.avg * 100:.2f}%"
        })
        
    return {
        "loss": loss_meter.avg, 
        "acc": acc_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion, 
    device: str, 
    use_amp: bool, 
    epoch: int, 
    total_epochs: int,
):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    progress = tqdm(loader, dynamic_ncols=True, desc=f"Eval {epoch:03d}/{total_epochs:03d}")
    
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
            
        batch_size = images.size(0)
        batch_acc = accuracy_from_logits(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_acc, batch_size)
        
        progress.set_postfix({
            "loss": f"{loss_meter.avg:.4f}", 
            "acc": f"{acc_meter.avg * 100:.2f}%"
        })
        
    return {
        "loss": loss_meter.avg, 
        "acc": acc_meter.avg,
    }


def save_checkpoint(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def main(cfg: TrainConfig):
    if cfg.dataset_name.lower() == "imagenet":
        raise ValueError(
            "Do not use train_source_model.py for ImageNet. "
            "Use torchvision pretrained weights for ImageNet to stay closest to the original repo."
        )
        
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    
    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)
    
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))
    
    best_acc = -1.0
    history = []
    
    dataset_info = {
        "dataset_name": cfg.dataset_name, 
        "data_root": cfg.data_root, 
        "model_name": cfg.model_name, 
        "num_classes": cfg.num_classes, 
        "class_names": getattr(train_ds, "classes", None), 
        "image_size": cfg.image_size, 
        "in_channels": cfg.in_channels, 
        "train_samples": len(train_ds), 
        "test_samples": len(test_ds), 
        "note": "This checkpoint is intended to serve as the clean source model for the attack pipeline."
    }
    
    with open(os.path.join(cfg.save_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
        
    print("=" * 100)
    print("Training clean source classifier for attack pipeline")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print("=" * 100)
    
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            scaler=scaler, 
            use_amp=use_amp, 
            epoch=epoch, 
            total_epochs=cfg.epochs,
        )
        
        eval_stats = evaluate(
            model=model, 
            loader=test_loader, 
            criterion=criterion, 
            device=device, 
            use_amp=use_amp, 
            epoch=epoch, 
            total_epochs=cfg.epochs,
        )
        
        if scheduler is not None:
            scheduler.step()
            
        lr_now = optimizer.param_groups[0]["lr"]
        
        row = {
            "epoch": epoch, 
            "lr": lr_now, 
            "train_loss": train_stats["loss"], 
            "train_acc": train_stats["acc"], 
            "val_loss": eval_stats["loss"], 
            "val_acc": eval_stats["acc"],
        }
        history.append(row)
        
        print(
            f"Epoch [{epoch:03d}/{cfg.epochs:03d}] "
            f"lr={lr_now:.6f} | "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc'] * 100:.2f}% | "
            f"val_loss={eval_stats['loss']:.4f} val_acc={eval_stats['acc'] * 100:.2f}%"
        )
        
        is_best = eval_stats["acc"] > best_acc
        if is_best:
            best_acc = eval_stats["acc"]
            
        checkpoint_payload = {
            "epoch": epoch,
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_acc": best_acc,
            "config": asdict(cfg),
            "history": history,
            "class_names": getattr(train_ds, "classes", None),
        }
        
        if not cfg.save_best_only:
            save_checkpoint(os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pth"), checkpoint_payload)
            
        if is_best:
            save_checkpoint(os.path.join(cfg.save_dir, "best_source_model.pth"), checkpoint_payload)
            
        with open(os.path.join(cfg.save_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)
            
        with open(os.path.join(cfg.save_dir, "train_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
            
        summary = {
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "best_val_acc": best_acc,
            "best_checkpoint": os.path.join(cfg.save_dir, "best_source_model.pth"),
        }
        
        with open(os.path.join(cfg.save_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
    print("=" * 100)
    print(f"Done. Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"Saved best checkpoint to: {os.path.join(cfg.save_dir, 'best_source_model.pth')}")
    print("=" * 100)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train clean source classifier for attack pipeline")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/source_model")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--milestones", type=int, nargs="*", default=[60, 80])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_best_only", action="store_true")
    
    args = parser.parse_args()
    
    return TrainConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        save_dir=args.save_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        in_channels=args.in_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        scheduler=args.scheduler,
        milestones=tuple(args.milestones),
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
        save_best_only=args.save_best_only,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)