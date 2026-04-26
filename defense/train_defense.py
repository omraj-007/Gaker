from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from defense.path_setup import add_project_root_to_path
add_project_root_to_path()

from utils.source_model_utils import SourceModelConfig, build_source_model, normalize_imagenet  # noqa: E402
from defense.datasets import DefenseDataset, defense_collate  # noqa: E402
from defense.losses import LossWeights, compute_defense_losses, label_accuracy, target_hit_rate  # noqa: E402
from defense.model import DefenseConfig, LatentSemanticDefense  # noqa: E402
from defense.utils import AverageMeter, ensure_dir, save_json, set_seed  # noqa: E402


@dataclass
class TrainDefenseConfig:
    dataset_root: str = "./defense/dataset"
    train_split: str = "train"
    save_dir: str = "./defense/checkpoints/baseline"

    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True

    source_model_name: str = "resnet50"
    source_model_checkpoint: Optional[str] = None
    source_model_num_classes: int = 1000
    use_imagenet_pretrained: bool = True

    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    purifier_blocks: int = 4

    lambda_pix: float = 1.0
    lambda_src_feat: float = 1.0
    lambda_cls: float = 0.25
    lambda_anti_tgt: float = 0.0

    save_example_batches: int = 2


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_train_loader(cfg: TrainDefenseConfig) -> DataLoader:
    train_ds = DefenseDataset(
        split_root=str(Path(cfg.dataset_root) / cfg.train_split),
        image_size=cfg.image_size,
        return_target=float(cfg.lambda_anti_tgt) > 0.0,
    )
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=defense_collate,
    )


def save_batch_visuals(
    adv: torch.Tensor,
    source: torch.Tensor,
    purified: torch.Tensor,
    save_path: str,
    max_items: int = 6,
) -> None:
    n = min(max_items, adv.size(0))
    grid = make_grid(torch.cat([adv[:n].cpu(), source[:n].cpu(), purified[:n].cpu()], dim=0), nrow=n, padding=2)
    save_image(grid, save_path)


def run_train_epoch(
    model: LatentSemanticDefense,
    source_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    weights: LossWeights,
    device: str,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    save_example_dir: Optional[str] = None,
    save_example_batches: int = 0,
) -> Dict[str, float]:
    model.train(True)
    if hasattr(source_model, "eval"):
        source_model.eval()

    meters = {
        "loss_total": AverageMeter(),
        "loss_pix": AverageMeter(),
        "loss_src_feat": AverageMeter(),
        "loss_cls": AverageMeter(),
        "source_acc_on_adv": AverageMeter(),
        "source_acc_on_purified": AverageMeter(),
        "target_hit_on_adv": AverageMeter(),
        "target_hit_on_purified": AverageMeter(),
    }
    if float(weights.lambda_anti_tgt) > 0.0:
        meters["loss_anti_tgt"] = AverageMeter()

    progress = tqdm(loader, dynamic_ncols=True, desc=f"train {epoch:03d}/{total_epochs:03d}")
    saved_example_count = 0

    for batch_idx, batch in enumerate(progress):
        adv = batch["adv"].to(device, non_blocking=True)
        source = batch["source"].to(device, non_blocking=True)
        source_labels = batch["source_label"].to(device, non_blocking=True)
        target_labels = batch["target_label"].to(device, non_blocking=True)
        target = batch["target"]
        if target is not None:
            target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(adv)
            x_hat = outputs["x_hat"]

            adv_logits = source_model.forward_logits(normalize_imagenet(adv))
            purified_logits = source_model.forward_logits(normalize_imagenet(x_hat))
            source_logits = source_model.forward_logits(normalize_imagenet(source))

            source_features = source_model.forward_features(normalize_imagenet(source))
            purified_features = source_model.forward_features(normalize_imagenet(x_hat))
            target_features = None
            if target is not None and float(weights.lambda_anti_tgt) > 0.0:
                target_features = source_model.forward_features(normalize_imagenet(target))

            loss_dict = compute_defense_losses(
                x_hat=x_hat,
                x_source=source,
                source_logits=source_logits,
                purified_logits=purified_logits,
                source_features=source_features,
                purified_features=purified_features,
                source_labels=source_labels,
                weights=weights,
                target_features=target_features,
            )

        scaler.scale(loss_dict["loss_total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = adv.size(0)
        for key in ["loss_total", "loss_pix", "loss_src_feat", "loss_cls"]:
            meters[key].update(loss_dict[key].item(), batch_size)
        if "loss_anti_tgt" in loss_dict:
            meters["loss_anti_tgt"].update(loss_dict["loss_anti_tgt"].item(), batch_size)

        adv_src_acc = label_accuracy(adv_logits.detach(), source_labels)
        purified_src_acc = label_accuracy(purified_logits.detach(), source_labels)
        adv_tgt_hit = target_hit_rate(adv_logits.detach(), target_labels)
        purified_tgt_hit = target_hit_rate(purified_logits.detach(), target_labels)

        meters["source_acc_on_adv"].update(adv_src_acc, batch_size)
        meters["source_acc_on_purified"].update(purified_src_acc, batch_size)
        meters["target_hit_on_adv"].update(adv_tgt_hit, batch_size)
        meters["target_hit_on_purified"].update(purified_tgt_hit, batch_size)

        progress.set_postfix({
            "loss": f"{meters['loss_total'].avg:.4f}",
            "src_adv": f"{meters['source_acc_on_adv'].avg * 100:.2f}%",
            "src_hat": f"{meters['source_acc_on_purified'].avg * 100:.2f}%",
            "tgt_adv": f"{meters['target_hit_on_adv'].avg * 100:.2f}%",
            "tgt_hat": f"{meters['target_hit_on_purified'].avg * 100:.2f}%",
        })

        if save_example_dir is not None and saved_example_count < save_example_batches:
            ensure_dir(save_example_dir)
            save_batch_visuals(
                adv=adv,
                source=source,
                purified=x_hat,
                save_path=os.path.join(save_example_dir, f"train_epoch_{epoch:03d}_batch_{batch_idx:03d}.png"),
            )
            saved_example_count += 1

    return {k: v.avg for k, v in meters.items()}


def main(cfg: TrainDefenseConfig) -> None:
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))

    train_loader = build_train_loader(cfg)

    model = LatentSemanticDefense(
        DefenseConfig(
            in_channels=3,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
            purifier_blocks=cfg.purifier_blocks,
        )
    ).to(device)

    source_model_cfg = SourceModelConfig(
        model_name=cfg.source_model_name,
        num_classes=cfg.source_model_num_classes,
        checkpoint_path=cfg.source_model_checkpoint,
        device=device,
        use_imagenet_pretrained=cfg.use_imagenet_pretrained,
        freeze=True,
    )
    source_model, source_model_meta = build_source_model(source_model_cfg)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    weights = LossWeights(
        lambda_pix=cfg.lambda_pix,
        lambda_src_feat=cfg.lambda_src_feat,
        lambda_cls=cfg.lambda_cls,
        lambda_anti_tgt=cfg.lambda_anti_tgt,
    )

    save_json(os.path.join(cfg.save_dir, "train_config.json"), asdict(cfg))

    best_train_loss = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_train_epoch(
            model=model,
            source_model=source_model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            weights=weights,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=cfg.epochs,
            save_example_dir=os.path.join(cfg.save_dir, "examples"),
            save_example_batches=cfg.save_example_batches,
        )

        row = {
            "epoch": epoch,
            "train": train_stats,
        }
        history.append(row)
        save_json(os.path.join(cfg.save_dir, "history.json"), {"history": history})

        current_train_loss = float(train_stats["loss_total"])
        if current_train_loss < best_train_loss:
            best_train_loss = current_train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "best_train_loss": best_train_loss,
                    "selection_rule": "lowest_train_loss",
                    "source_model_meta_keys": list(source_model_meta.keys()) if isinstance(source_model_meta, dict) else [],
                },
                os.path.join(cfg.save_dir, "best_defense.pt"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "best_train_loss": best_train_loss,
                "selection_rule": "lowest_train_loss",
            },
            os.path.join(cfg.save_dir, "last_defense.pt"),
        )

        print(json.dumps(row, indent=2))

    save_json(
        os.path.join(cfg.save_dir, "train_summary.json"),
        {
            "best_train_loss": best_train_loss,
            "selection_rule": "lowest_train_loss",
            "best_checkpoint": os.path.join(cfg.save_dir, "best_defense.pt"),
            "last_checkpoint": os.path.join(cfg.save_dir, "last_defense.pt"),
            "note": "No test-set evaluation is performed in train_defense.py. Use evaluate_defense.py separately on defense/dataset/test.",
        },
    )


def parse_args() -> TrainDefenseConfig:
    parser = argparse.ArgumentParser(description="Train latent-semantic purification defense on train split only")
    parser.add_argument("--dataset_root", type=str, default="./defense/dataset")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--save_dir", type=str, default="./defense/checkpoints/baseline")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--source_model_name", type=str, default="resnet50")
    parser.add_argument("--source_model_checkpoint", type=str, default="")
    parser.add_argument("--source_model_num_classes", type=int, default=1000)
    parser.add_argument("--use_imagenet_pretrained", type=str2bool, default=True)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_mults", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--purifier_blocks", type=int, default=4)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_src_feat", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.25)
    parser.add_argument("--lambda_anti_tgt", type=float, default=0.0)
    parser.add_argument("--save_example_batches", type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = args.source_model_checkpoint.strip()
    return TrainDefenseConfig(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        save_dir=args.save_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
        source_model_name=args.source_model_name,
        source_model_checkpoint=checkpoint_path if checkpoint_path else None,
        source_model_num_classes=args.source_model_num_classes,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        purifier_blocks=args.purifier_blocks,
        lambda_pix=args.lambda_pix,
        lambda_src_feat=args.lambda_src_feat,
        lambda_cls=args.lambda_cls,
        lambda_anti_tgt=args.lambda_anti_tgt,
        save_example_batches=args.save_example_batches,
    )


if __name__ == "__main__":
    main(parse_args())
