from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw
from torchvision.utils import save_image


def load_success_records(metadata_path: Path) -> List[Dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    records: List[Dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if bool(obj.get("targeted_success", False)):
                records.append(obj)
    return records


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pad_to_square(img: Image.Image, side: int, bg=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (side, side), bg)
    x = (side - img.size[0]) // 2
    y = (side - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def load_panel_image(export_root: Path, rel_path: str, panel_size: int) -> Image.Image:
    img_path = export_root / rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image referenced in metadata not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    img = resize_keep_aspect(img, panel_size)
    img = pad_to_square(img, panel_size)
    return img


def build_grid(
    export_root: Path,
    records: List[Dict],
    output_path: Path,
    panel_size: int = 160,
    margin: int = 16,
    label_height: int = 22,
    row_gap: int = 18,
) -> None:
    n = len(records)
    if n == 0:
        raise ValueError("No successful records to visualize.")

    cols = 3
    width = margin * (cols + 1) + panel_size * cols
    row_h = label_height * 2 + panel_size + row_gap
    height = margin + n * row_h + label_height

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin + 35, 6), "Source", fill=(0, 0, 0))
    draw.text((margin * 2 + panel_size + 35, 6), "Target", fill=(0, 0, 0))
    draw.text((margin * 3 + panel_size * 2 + 25, 6), "Perturbed", fill=(0, 0, 0))

    for row_idx, rec in enumerate(records):
        y0 = margin + row_idx * row_h + label_height

        source_img = load_panel_image(export_root, rec["source_image"], panel_size)
        target_img = load_panel_image(export_root, rec["target_image"], panel_size)
        adv_img = load_panel_image(export_root, rec["adv_image"], panel_size)

        x1 = margin
        x2 = margin * 2 + panel_size
        x3 = margin * 3 + panel_size * 2

        canvas.paste(source_img, (x1, y0))
        canvas.paste(target_img, (x2, y0))
        canvas.paste(adv_img, (x3, y0))

        info = (
            f"id={rec.get('sample_id')} | "
            f"src={rec.get('source_label')} -> tgt={rec.get('target_label')} | "
            f"pred={rec.get('adv_pred_label')}"
        )
        draw.text((margin, y0 + panel_size + 4), info, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_individual_triptychs(
    export_root: Path,
    records: List[Dict],
    out_dir: Path,
    panel_size: int = 256,
    margin: int = 16,
    label_height: int = 22,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        width = panel_size * 3 + margin * 4
        height = panel_size + label_height * 2 + margin * 2
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        source_img = load_panel_image(export_root, rec["source_image"], panel_size)
        target_img = load_panel_image(export_root, rec["target_image"], panel_size)
        adv_img = load_panel_image(export_root, rec["adv_image"], panel_size)

        x1 = margin
        x2 = margin * 2 + panel_size
        x3 = margin * 3 + panel_size * 2
        y = margin + label_height

        canvas.paste(source_img, (x1, y))
        canvas.paste(target_img, (x2, y))
        canvas.paste(adv_img, (x3, y))

        draw.text((x1 + 12, margin), "Source", fill=(0, 0, 0))
        draw.text((x2 + 12, margin), "Target", fill=(0, 0, 0))
        draw.text((x3 + 12, margin), "Perturbed", fill=(0, 0, 0))

        info = (
            f"sample_id={rec.get('sample_id')} | "
            f"source_label={rec.get('source_label')} | "
            f"target_label={rec.get('target_label')} | "
            f"adv_pred_label={rec.get('adv_pred_label')}"
        )
        draw.text((margin, y + panel_size + 4), info, fill=(0, 0, 0))

        stem = f"sample_{int(rec.get('sample_id', 0)):06d}.png"
        canvas.save(out_dir / stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show 10 successful source/target/perturbed examples from an exported adversarial dataset."
    )
    parser.add_argument(
        "--export_root",
        type=str,
        required=True,
        help="Path to exported adversarial dataset folder containing metadata.jsonl and source/target/adv subfolders.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of successful examples to visualize.",
    )
    parser.add_argument(
        "--random_select",
        action="store_true",
        help="Randomly pick successful examples instead of taking the first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when --random_select is used.",
    )
    parser.add_argument(
        "--output_grid",
        type=str,
        default="successful_examples_grid.png",
        help="Output filename for the combined grid image.",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Also save one triptych image per selected successful sample.",
    )
    parser.add_argument(
        "--individual_dir",
        type=str,
        default="successful_examples_individual",
        help="Folder name for individual triptychs (inside export_root unless absolute path is given).",
    )
    args = parser.parse_args()

    export_root = Path(args.export_root)
    metadata_path = export_root / "metadata.jsonl"

    success_records = load_success_records(metadata_path)
    if len(success_records) == 0:
        raise ValueError(f"No successful examples found in {metadata_path}")

    n = min(args.num_images, len(success_records))
    if args.random_select:
        rng = random.Random(args.seed)
        selected = rng.sample(success_records, n)
    else:
        selected = success_records[:n]

    output_grid = Path(args.output_grid)
    if not output_grid.is_absolute():
        output_grid = export_root / output_grid

    build_grid(
        export_root=export_root,
        records=selected,
        output_path=output_grid,
        panel_size=160,
    )

    if args.save_individual:
        individual_dir = Path(args.individual_dir)
        if not individual_dir.is_absolute():
            individual_dir = export_root / individual_dir
        save_individual_triptychs(
            export_root=export_root,
            records=selected,
            out_dir=individual_dir,
            panel_size=256,
        )

    print("=" * 80)
    print(f"Successful examples available: {len(success_records)}")
    print(f"Selected examples:            {n}")
    print(f"Saved combined grid to:       {output_grid}")
    if args.save_individual:
        print(f"Saved individual triptychs to:{individual_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
