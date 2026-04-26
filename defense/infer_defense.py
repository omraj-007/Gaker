from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from defense.model import DefenseConfig, LatentSemanticDefense
from defense.utils import ensure_dir


@dataclass
class InferConfig:
    checkpoint_path: str
    input_image: str = ""
    input_dir: str = ""
    output_dir: str = "./defense/inference_outputs"
    image_size: int = 224
    device: str = "cuda"
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    purifier_blocks: int = 4


def collect_input_paths(cfg: InferConfig) -> List[Path]:
    if cfg.input_image:
        return [Path(cfg.input_image)]
    if cfg.input_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return sorted([p for p in Path(cfg.input_dir).iterdir() if p.is_file() and p.suffix.lower() in exts])
    raise ValueError("Provide either --input_image or --input_dir")


def main(cfg: InferConfig) -> None:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg.output_dir)

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

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    paths = collect_input_paths(cfg)
    with torch.no_grad():
        for path in tqdm(paths, desc="Purifying"):
            image = Image.open(path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            x_hat = model(x)["x_hat"]
            save_image(x_hat[0].cpu(), str(Path(cfg.output_dir) / path.name))


def parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(description="Inference-only purification: input only x_adv")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--input_image", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./defense/inference_outputs")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_mults", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--purifier_blocks", type=int, default=4)
    args = parser.parse_args()
    return InferConfig(
        checkpoint_path=args.checkpoint_path,
        input_image=args.input_image,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        device=args.device,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        purifier_blocks=args.purifier_blocks,
    )


if __name__ == "__main__":
    main(parse_args())
