from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from defense.utils import ensure_dir, read_jsonl, save_json, write_jsonl


SUBFOLDERS = ["adv", "source", "target", "delta_vis", "delta_tensor"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _copy_or_link(src: Path, dst: Path, use_symlink: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _rewrite_record(record: Dict, split_root: Path, original_export_root: Path) -> Dict:
    new_record = dict(record)
    for key in ["source_image", "target_image", "adv_image", "delta_visualization", "delta_tensor"]:
        rel = new_record.get(key)
        if rel is None:
            continue
        rel_path = Path(rel)
        if rel_path.parts[0] not in SUBFOLDERS:
            raise ValueError(f"Unexpected relative path in metadata for {key}: {rel}")
        new_record[key] = str(rel_path)
    new_record["original_export_root"] = str(original_export_root)
    new_record["split_root"] = str(split_root)
    return new_record


def build_split(
    export_root: str,
    output_root: str,
    train_ratio: float = 0.7,
    seed: int = 42,
    use_symlink: bool = True,
) -> Dict:
    export_root_path = Path(export_root)
    output_root_path = Path(output_root)
    metadata_path = export_root_path / "metadata.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    records = read_jsonl(metadata_path)
    if len(records) == 0:
        raise ValueError(f"No records found in {metadata_path}")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    num_train = int(round(len(shuffled) * float(train_ratio)))
    num_train = max(1, min(len(shuffled) - 1, num_train))
    train_records = shuffled[:num_train]
    test_records = shuffled[num_train:]

    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        split_root = output_root_path / split_name
        ensure_dir(split_root)
        for subfolder in SUBFOLDERS:
            ensure_dir(split_root / subfolder)

        rewritten: List[Dict] = []
        for record in split_records:
            for meta_key in [
                ("source_image", "source"),
                ("target_image", "target"),
                ("adv_image", "adv"),
                ("delta_visualization", "delta_vis"),
                ("delta_tensor", "delta_tensor"),
            ]:
                key, expected_prefix = meta_key
                rel = record.get(key)
                if rel is None:
                    continue
                src = export_root_path / rel
                if not src.exists():
                    raise FileNotFoundError(f"Referenced file missing for {key}: {src}")
                dst = split_root / rel
                if Path(rel).parts[0] != expected_prefix:
                    raise ValueError(f"Expected {key} to start with {expected_prefix}, got {rel}")
                _copy_or_link(src=src, dst=dst, use_symlink=use_symlink)

            rewritten.append(_rewrite_record(record=record, split_root=split_root, original_export_root=export_root_path))

        write_jsonl(split_root / "metadata.jsonl", rewritten)
        save_json(
            split_root / "split_summary.json",
            {
                "split": split_name,
                "num_samples": len(rewritten),
                "source_export_root": str(export_root_path),
                "use_symlink": bool(use_symlink),
            },
        )

    summary = {
        "export_root": str(export_root_path),
        "output_root": str(output_root_path),
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "use_symlink": bool(use_symlink),
        "num_total": len(records),
        "num_train": len(train_records),
        "num_test": len(test_records),
    }
    save_json(output_root_path / "dataset_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 70/30 defense train-test split from exported adversarial dataset")
    parser.add_argument("--export_root", type=str, required=True, help="Path to exported_adv/... folder containing metadata.jsonl")
    parser.add_argument("--output_root", type=str, default="./defense/dataset", help="Output path for defense dataset split")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--use_symlink", type=str2bool, default=True, help="Use symlinks instead of copying files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_split(
        export_root=args.export_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        use_symlink=args.use_symlink,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
