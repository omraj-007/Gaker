from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from defense.path_setup import add_project_root_to_path
add_project_root_to_path()

from utils.source_model_utils import SourceModelConfig, build_source_model, normalize_imagenet  # noqa: E402
from defense.datasets import DefenseDataset, defense_collate  # noqa: E402
from defense.losses import label_accuracy, target_hit_rate  # noqa: E402
from defense.model import DefenseConfig, LatentSemanticDefense  # noqa: E402
from defense.utils import AverageMeter, ensure_dir, save_json  # noqa: E402


@dataclass
class EvalDefenseConfig:
    split_root: str = "./defense/dataset/test"
    checkpoint_path: str = "./defense/checkpoints/baseline/best_defense.pt"
    save_dir: str = "./defense/eval_outputs"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    use_amp: bool = True

    source_model_name: str = "resnet50"
    source_model_checkpoint: Optional[str] = None
    source_model_num_classes: int = 1000
    use_imagenet_pretrained: bool = True

    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    purifier_blocks: int = 4


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def main(cfg: EvalDefenseConfig) -> None:
    ensure_dir(cfg.save_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))

    dataset = DefenseDataset(split_root=cfg.split_root, image_size=cfg.image_size, return_target=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=defense_collate,
    )

    model = LatentSemanticDefense(
        DefenseConfig(
            in_channels=3,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
            purifier_blocks=cfg.purifier_blocks,
        )
    ).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    source_model_cfg = SourceModelConfig(
        model_name=cfg.source_model_name,
        num_classes=cfg.source_model_num_classes,
        checkpoint_path=cfg.source_model_checkpoint,
        device=device,
        use_imagenet_pretrained=cfg.use_imagenet_pretrained,
        freeze=True,
    )
    source_model, _ = build_source_model(source_model_cfg)
    source_model.eval()

    meters = {
        "source_acc_on_adv": AverageMeter(),
        "source_acc_on_purified": AverageMeter(),
        "target_hit_on_adv": AverageMeter(),
        "target_hit_on_purified": AverageMeter(),
    }

    with torch.no_grad():
        progress = tqdm(loader, dynamic_ncols=True, desc="Evaluate defense")
        for batch in progress:
            adv = batch["adv"].to(device, non_blocking=True)
            source_labels = batch["source_label"].to(device, non_blocking=True)
            target_labels = batch["target_label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_hat = model(adv)["x_hat"]
                adv_logits = source_model.forward_logits(normalize_imagenet(adv))
                hat_logits = source_model.forward_logits(normalize_imagenet(x_hat))

            batch_size = adv.size(0)
            src_adv = label_accuracy(adv_logits, source_labels)
            src_hat = label_accuracy(hat_logits, source_labels)
            tgt_adv = target_hit_rate(adv_logits, target_labels)
            tgt_hat = target_hit_rate(hat_logits, target_labels)

            meters["source_acc_on_adv"].update(src_adv, batch_size)
            meters["source_acc_on_purified"].update(src_hat, batch_size)
            meters["target_hit_on_adv"].update(tgt_adv, batch_size)
            meters["target_hit_on_purified"].update(tgt_hat, batch_size)

            progress.set_postfix({
                "src_adv": f"{meters['source_acc_on_adv'].avg * 100:.2f}%",
                "src_hat": f"{meters['source_acc_on_purified'].avg * 100:.2f}%",
                "tgt_adv": f"{meters['target_hit_on_adv'].avg * 100:.2f}%",
                "tgt_hat": f"{meters['target_hit_on_purified'].avg * 100:.2f}%",
            })

    summary = {k: v.avg for k, v in meters.items()}
    summary["checkpoint_path"] = cfg.checkpoint_path
    summary["split_root"] = cfg.split_root
    save_json(Path(cfg.save_dir) / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_args() -> EvalDefenseConfig:
    parser = argparse.ArgumentParser(description="Evaluate trained defense on a split")
    parser.add_argument("--split_root", type=str, default="./defense/dataset/test")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./defense/eval_outputs")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--source_model_name", type=str, default="resnet50")
    parser.add_argument("--source_model_checkpoint", type=str, default="")
    parser.add_argument("--source_model_num_classes", type=int, default=1000)
    parser.add_argument("--use_imagenet_pretrained", type=str2bool, default=True)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_mults", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--purifier_blocks", type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = args.source_model_checkpoint.strip()
    return EvalDefenseConfig(
        split_root=args.split_root,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_amp=args.use_amp,
        source_model_name=args.source_model_name,
        source_model_checkpoint=checkpoint_path if checkpoint_path else None,
        source_model_num_classes=args.source_model_num_classes,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        purifier_blocks=args.purifier_blocks,
    )


if __name__ == "__main__":
    main(parse_args())
