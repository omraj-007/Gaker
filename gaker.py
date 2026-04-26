from generator.train import train
from generator.craftadv import craftadv

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def infer_dataset_defaults(dataset_name: str):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        return {
            "num_classes": 10,
            "image_size": 32,
            "scale_size": 32,
            "train_subdir": "cifar-10-batches-py",
            "val_subdir": "cifar-10-batches-py",
        }

    if dataset_name == "cifar100":
        return {
            "num_classes": 100,
            "image_size": 32,
            "scale_size": 32,
            "train_subdir": "cifar-100-python",
            "val_subdir": "cifar-100-python",
        }

    if dataset_name == "tinyimagenet":
        return {
            "num_classes": 200,
            "image_size": 64,
            "scale_size": 64,
            "train_subdir": "tiny-imagenet-200",
            "val_subdir": "tiny-imagenet-200",
        }

    if dataset_name == "imagenet":
        return {
            "num_classes": 1000,
            "image_size": 224,
            "scale_size": 256,
            "train_subdir": "imagenet_data",
            "val_subdir": "imagenet_data",
        }

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_parser():
    parser = argparse.ArgumentParser(description="Configurable GAKer entry point")

    # mode
    parser.add_argument(
        "--state",
        type=str,
        default="train_model",
        choices=["train_model", "craftadv"],
        help="Run training or adversarial crafting",
    )

    # dataset / paths
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tinyimagenet",
        choices=["cifar10", "cifar100", "tinyimagenet", "imagenet"],
        help="Dataset name",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="./datasets",
        help="Root folder that contains all datasets",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="",
        help="Optional explicit training directory override",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="",
        help="Optional explicit target directory override",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="",
        help="Optional explicit validation/source directory override for craftadv",
    )

    # source feature extractor
    parser.add_argument(
        "--Source_Model",
        type=str,
        default="ResNet50",
        choices=["ResNet18", "ResNet50", "DenseNet121"],
        help="Frozen source model used for target feature extraction",
    )
    parser.add_argument(
        "--use_pretrained",
        type=str2bool,
        default=True,
        help="Use torchvision pretrained weights for the source model",
    )

    # generator hyperparameters
    parser.add_argument("--epoch", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--channel", type=int, default=32, help="Base channel count")
    parser.add_argument(
        "--channel_mult",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Channel multipliers for the original generator",
    )
    parser.add_argument("--num_res_blocks", type=int, default=1, help="Number of residual blocks")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps", type=float, default=16.0 / 255.0, help="Linf epsilon")
    parser.add_argument("--noise_loss_weight", type=float, default=0.5, help="Weight for noise-feature loss")

    # image processing
    parser.add_argument(
        "--image_size",
        type=int,
        default=-1,
        help="Input crop size. Use -1 to adopt dataset default",
    )
    parser.add_argument(
        "--scale_size",
        type=int,
        default=-1,
        help="Resize size before crop. Use -1 to adopt dataset default",
    )

    # target protocol
    parser.add_argument(
        "--set_targets",
        type=str,
        default="all_classes",
        help="Target protocol. Use all_classes for CIFAR/TinyImageNet. Keep targets_200 for original ImageNet setup only.",
    )
    parser.add_argument(
        "--unknown",
        type=str2bool,
        default=False,
        help="Whether to craft on unknown target setting",
    )
    parser.add_argument(
        "--target_select",
        type=str,
        default="1",
        choices=["1", "10"],
        help="Number of target images to sample per class during craftadv",
    )
    parser.add_argument(
        "--ran_best",
        type=str,
        default="random",
        choices=["random", "best"],
        help="Target feature selection mode used in craftadv",
    )

    # runtime
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # saving
    parser.add_argument(
        "--Generator_save_dir",
        type=str,
        default="./checkpoints/gaker_runs",
        help="Directory to save generator checkpoints and metadata",
    )
    parser.add_argument(
        "--test_load_weight",
        type=str,
        default="",
        help="Checkpoint filename for craftadv",
    )
    parser.add_argument(
        "--save_run_metadata",
        type=str2bool,
        default=True,
        help="Save config and runtime metadata",
    )
    parser.add_argument('--num_known_classes', type=int, default=200, help='Number of known classes for random split')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for class split')
    parser.add_argument('--samples_per_known_class', type=int, default=325, help='Number of images per known class')
    
    return parser


def resolve_paths(args):
    ds_defaults = infer_dataset_defaults(args.dataset_name)

    image_size = ds_defaults["image_size"] if args.image_size == -1 else args.image_size
    scale_size = ds_defaults["scale_size"] if args.scale_size == -1 else args.scale_size

    dataset_root = os.path.join(args.datasets_root, ds_defaults["train_subdir"])

    train_dir = args.train_dir if args.train_dir else dataset_root
    target_dir = args.target_dir if args.target_dir else dataset_root
    val_dir = args.val_dir if args.val_dir else dataset_root

    return {
        "num_classes": ds_defaults["num_classes"],
        "image_size": image_size,
        "scale_size": scale_size,
        "dataset_root": dataset_root,
        "train_dir": train_dir,
        "target_dir": target_dir,
        "val_dir": val_dir,
    }


def save_metadata(modelConfig):
    save_dir = modelConfig["Generator_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    metadata = dict(modelConfig)
    metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = build_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    resolved = resolve_paths(args)

    modelConfig = {
        "state": args.state,
        "dataset_name": args.dataset_name.lower(),
        "datasets_root": args.datasets_root,
        "dataset_root": resolved["dataset_root"],
        "train_dir": resolved["train_dir"],
        "target_dir": resolved["target_dir"],
        "val_dir": resolved["val_dir"],
        "num_classes": resolved["num_classes"],
        "image_size": resolved["image_size"],
        "scale_size": resolved["scale_size"],
        "Source_Model": args.Source_Model,
        "use_pretrained": args.use_pretrained,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "lr": args.lr,
        "eps": args.eps,
        "noise_loss_weight": args.noise_loss_weight,
        "device": args.device,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "test_load_weight": args.test_load_weight,
        "Generator_save_dir": args.Generator_save_dir,
        "set_targets": args.set_targets,
        "unknown": args.unknown,
        "target_select": args.target_select,
        "ran_best": args.ran_best,
        "save_run_metadata": args.save_run_metadata,
        'num_known_classes': args.num_known_classes,
        'split_seed': args.split_seed,
        'samples_per_known_class': args.samples_per_known_class,
    }

    os.makedirs(modelConfig["Generator_save_dir"], exist_ok=True)

    if modelConfig["save_run_metadata"]:
        save_metadata(modelConfig)

    print("=" * 100)
    print("Resolved configuration")
    print(json.dumps(modelConfig, indent=2))
    print("=" * 100)

    if modelConfig["state"] == "train_model":
        train(modelConfig)
    elif modelConfig["state"] == "craftadv":
        if not modelConfig["test_load_weight"]:
            raise ValueError("--test_load_weight is required for craftadv")
        craftadv(modelConfig)


if __name__ == "__main__":
    main()