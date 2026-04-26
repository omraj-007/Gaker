import os
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from generator import Generator
from utils.gaussian_smoothing import get_gaussian_kernel


def normalize(t: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = t.clone()
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image, int(label), int(idx)


class TinyImageNetValDataset(Dataset):
    """
    Supports the official raw Tiny-ImageNet val layout:

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


class CustomDenseNet121(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class CustomResnet18(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CustomResnet50(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _load_resnet18(use_pretrained: bool):
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        return torchvision.models.resnet18(weights=weights)
    except AttributeError:
        return torchvision.models.resnet18(pretrained=use_pretrained)


def _load_resnet50(use_pretrained: bool):
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        return torchvision.models.resnet50(weights=weights)
    except AttributeError:
        return torchvision.models.resnet50(pretrained=use_pretrained)


def _load_densenet121(use_pretrained: bool):
    try:
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
        return torchvision.models.densenet121(weights=weights)
    except AttributeError:
        return torchvision.models.densenet121(pretrained=use_pretrained)


def build_feature_extractor(modelConfig: Dict, device: torch.device):
    model_name = modelConfig["Source_Model"]
    use_pretrained = bool(modelConfig.get("use_pretrained", True))

    if model_name == "ResNet18":
        original_model = _load_resnet18(use_pretrained)
        feature_extraction = CustomResnet18(original_model)
        feature_channel = 512

    elif model_name == "ResNet50":
        original_model = _load_resnet50(use_pretrained)
        feature_extraction = CustomResnet50(original_model)
        feature_channel = 2048

    elif model_name == "DenseNet121":
        original_model = _load_densenet121(use_pretrained)
        feature_extraction = CustomDenseNet121(original_model)
        feature_channel = 1024

    else:
        raise ValueError(
            f"Unsupported Source_Model: {model_name}. "
            f"Use ResNet18, ResNet50, or DenseNet121."
        )

    freeze_model(feature_extraction)
    feature_extraction = feature_extraction.to(device)
    return feature_extraction, feature_channel


def resolve_imagefolder_root(base_dir: str, candidate_subdirs: List[str]) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for subdir in candidate_subdirs:
        candidate = os.path.join(base_dir, subdir)
        if os.path.isdir(candidate):
            return candidate

    return base_dir


def get_transforms(modelConfig: Dict):
    img_size = int(modelConfig["image_size"])
    scale_size = int(modelConfig["scale_size"])

    source_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    return source_transform, target_transform


def build_cifar_dataset(dataset_name: str, root: str, train: bool, transform):
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    if dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    raise ValueError(f"Unsupported CIFAR dataset: {dataset_name}")


def build_imagefolder_dataset(dataset_name: str, root: str, split: str, transform):
    if dataset_name == "tinyimagenet":
        if split == "train":
            train_root = resolve_imagefolder_root(root, ["train"])
            return torchvision.datasets.ImageFolder(train_root, transform=transform)

        # prefer already-reorganized val_images, else support official raw val/
        val_images_root = os.path.join(root, "val_images")
        if os.path.isdir(val_images_root):
            return torchvision.datasets.ImageFolder(val_images_root, transform=transform)

        raw_val_root = os.path.join(root, "val")
        if os.path.isdir(raw_val_root):
            images_dir = os.path.join(raw_val_root, "images")
            ann_path = os.path.join(raw_val_root, "val_annotations.txt")
            if os.path.isdir(images_dir) and os.path.isfile(ann_path):
                return TinyImageNetValDataset(root=root, transform=transform)

        val_root = resolve_imagefolder_root(root, ["val"])
        return torchvision.datasets.ImageFolder(val_root, transform=transform)

    if dataset_name == "imagenet":
        if split == "train":
            train_root = resolve_imagefolder_root(root, ["train"])
            return torchvision.datasets.ImageFolder(train_root, transform=transform)
        val_root = resolve_imagefolder_root(root, ["val"])
        return torchvision.datasets.ImageFolder(val_root, transform=transform)

    raise ValueError(f"Unsupported ImageFolder dataset: {dataset_name}")


def get_dataset_labels(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        base_labels = get_dataset_labels(dataset.dataset)
        return [int(base_labels[i]) for i in dataset.indices]

    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))
    return labels


def get_dataset_item(dataset: Dataset, idx: int):
    if isinstance(dataset, Subset):
        base_idx = dataset.indices[idx]
        return get_dataset_item(dataset.dataset, base_idx)
    return dataset[idx]


def get_dataset_path(dataset: Dataset, idx: int) -> Optional[str]:
    if isinstance(dataset, Subset):
        base_idx = dataset.indices[idx]
        return get_dataset_path(dataset.dataset, base_idx)

    if hasattr(dataset, "samples"):
        return dataset.samples[idx][0]

    return None


def get_dataset_classes(dataset: Dataset) -> List[str]:
    if isinstance(dataset, Subset):
        return get_dataset_classes(dataset.dataset)
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    return []


def resolve_target_list(set_targets: str, num_classes: int) -> List[int]:
    if set_targets == "targets_200":
        if num_classes != 1000:
            raise ValueError(
                "set_targets='targets_200' is only valid for ImageNet-1K. "
                "Use set_targets='all_classes' for CIFAR-10 / CIFAR-100 / Tiny-ImageNet."
            )
        return [
            22, 30, 43, 51, 53, 67, 76, 84, 107, 111, 116, 139, 156, 163, 174, 191, 194, 199,
            228, 241, 251, 288, 301, 310, 313, 323, 324, 354, 393, 398, 399, 401, 405, 418, 419,
            420, 422, 428, 429, 439, 441, 451, 455, 457, 465, 467, 478, 480, 481, 488, 489, 490,
            493, 496, 498, 499, 500, 507, 508, 514, 515, 519, 523, 530, 532, 533, 539, 540, 550,
            552, 553, 557, 565, 566, 575, 576, 579, 583, 588, 592, 593, 594, 599, 601, 604, 605,
            606, 607, 608, 611, 614, 622, 627, 640, 644, 646, 647, 659, 660, 666, 668, 674, 678,
            683, 684, 687, 688, 691, 694, 700, 704, 712, 714, 715, 722, 726, 729, 738, 739, 740,
            741, 749, 751, 761, 766, 769, 772, 773, 783, 785, 789, 790, 793, 794, 796, 798, 800,
            807, 815, 822, 825, 826, 831, 843, 844, 851, 853, 854, 855, 858, 860, 862, 863, 869,
            876, 877, 879, 880, 884, 888, 891, 897, 898, 901, 903, 904, 908, 910, 912, 914, 916,
            918, 919, 924, 925, 927, 931, 932, 933, 934, 937, 938, 943, 946, 950, 952, 954, 958,
            959, 961, 963, 971, 974, 977, 979, 980, 984, 985, 995, 996
        ]

    if set_targets == "all_classes":
        return list(range(num_classes))

    raise ValueError(
        f"Unsupported set_targets: {set_targets}. Use 'all_classes' or 'targets_200'."
    )


def build_source_and_target_datasets(modelConfig: Dict, source_transform, target_transform):
    dataset_name = modelConfig["dataset_name"].lower()
    craft_split = modelConfig.get("craft_split", None)
    target_split = modelConfig.get("target_split", None)

    if dataset_name in {"cifar10", "cifar100"}:
        source_is_train = bool(craft_split == "train")
        target_is_train = True if target_split is None else bool(target_split == "train")

        source_dataset = build_cifar_dataset(
            dataset_name=dataset_name,
            root=modelConfig["datasets_root"],
            train=source_is_train,
            transform=source_transform,
        )
        target_dataset = build_cifar_dataset(
            dataset_name=dataset_name,
            root=modelConfig["datasets_root"],
            train=target_is_train,
            transform=target_transform,
        )

    elif dataset_name == "tinyimagenet":
        source_split = "val" if craft_split is None else craft_split
        target_split = "train" if target_split is None else target_split

        source_dataset = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["val_dir"],
            split=source_split,
            transform=source_transform,
        )
        target_dataset = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["target_dir"],
            split=target_split,
            transform=target_transform,
        )

    elif dataset_name == "imagenet":
        source_split = "val" if craft_split is None else craft_split
        target_split = "train" if target_split is None else target_split

        source_dataset = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["val_dir"],
            split=source_split,
            transform=source_transform,
        )
        target_dataset = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["target_dir"],
            split=target_split,
            transform=target_transform,
        )

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    class_names = get_dataset_classes(target_dataset)
    num_classes = len(class_names) if len(class_names) > 0 else int(modelConfig["num_classes"])
    targets = resolve_target_list(modelConfig["set_targets"], num_classes)

    if bool(modelConfig.get("unknown", False)):
        complement = [i for i in range(num_classes) if i not in targets]
        if len(complement) == 0:
            raise ValueError(
                "unknown=True produced an empty target set. "
                "This usually happens when set_targets='all_classes'."
            )
        targets = complement

    target_labels = get_dataset_labels(target_dataset)
    target_indices = [i for i, label in enumerate(target_labels) if int(label) in targets]
    target_dataset = Subset(target_dataset, target_indices)

    return source_dataset, target_dataset, class_names, targets


def build_target_pools(target_dataset: Dataset) -> Dict[int, List[int]]:
    target_labels = get_dataset_labels(target_dataset)
    pools = {}
    for idx, label in enumerate(target_labels):
        pools.setdefault(int(label), []).append(int(idx))
    return pools


def choose_target_indices_for_batch(
    source_labels: torch.Tensor,
    target_pools: Dict[int, List[int]],
    allowed_target_labels: List[int],
    target_select: str,
) -> Tuple[List[int], List[int]]:
    selected_target_dataset_indices = []
    selected_target_labels = []

    candidate_pool_size = 1 if str(target_select) == "1" else 10

    for src_label in source_labels.tolist():
        valid_target_labels = [
            t for t in allowed_target_labels
            if t != int(src_label) and t in target_pools and len(target_pools[t]) > 0
        ]

        if len(valid_target_labels) == 0:
            raise RuntimeError(
                f"No valid target class available for source label {src_label}. "
                f"Check target pools and target protocol."
            )

        chosen_target_label = random.choice(valid_target_labels)
        available_indices = target_pools[chosen_target_label]

        k = min(candidate_pool_size, len(available_indices))
        sampled_candidates = random.sample(available_indices, k=k)

        # keep random selection to stay close to original repo behavior
        chosen_target_dataset_idx = random.choice(sampled_candidates)

        selected_target_dataset_indices.append(int(chosen_target_dataset_idx))
        selected_target_labels.append(int(chosen_target_label))

    return selected_target_dataset_indices, selected_target_labels


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_delta_vis(delta: torch.Tensor, save_path: str, magnify: float = 8.0) -> None:
    delta_vis = torch.clamp(delta * magnify + 0.5, 0.0, 1.0)
    save_image(delta_vis, save_path)


def load_training_metadata_if_present(generator_save_dir: str) -> Dict:
    dataset_info_path = os.path.join(generator_save_dir, "dataset_info.json")
    if os.path.isfile(dataset_info_path):
        with open(dataset_info_path, "r") as f:
            return json.load(f)
    return {}


def craftadv(modelConfig: Dict):
    device = torch.device(modelConfig["device"] if torch.cuda.is_available() else "cpu")

    # use saved training metadata when available, but let explicit current config win
    saved_info = load_training_metadata_if_present(modelConfig["Generator_save_dir"])
    merged = dict(saved_info)
    merged.update(modelConfig)
    modelConfig = merged

    source_transform, target_transform = get_transforms(modelConfig)

    source_dataset, target_dataset, class_names, targets = build_source_and_target_datasets(
        modelConfig=modelConfig,
        source_transform=source_transform,
        target_transform=target_transform,
    )

    target_pools = build_target_pools(target_dataset)
    source_indexed_dataset = IndexedDataset(source_dataset)

    source_loader = DataLoader(
        source_indexed_dataset,
        batch_size=int(modelConfig["batch_size"]),
        shuffle=False,
        num_workers=int(modelConfig.get("num_workers", 12)),
        pin_memory=True,
        drop_last=False,
    )

    feature_extraction, feature_channel = build_feature_extractor(
        modelConfig=modelConfig,
        device=device,
    )

    generator = Generator(
        num_target=len(targets),
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        feature_channel_num=feature_channel,
    ).to(device)

    ckpt_path = os.path.join(modelConfig["Generator_save_dir"], modelConfig["test_load_weight"])
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt, strict=False)
    print("model load weight done.")

    ran_best = str(modelConfig.get("ran_best", "random")).lower()
    if ran_best != "random":
        raise NotImplementedError(
            "ran_best != 'random' is not implemented in this generalized pipeline. "
            "Use --ran_best random."
        )

    generator.eval()
    freeze_model(generator)

    eps = float(modelConfig.get("eps", 16.0 / 255.0))
    print("eps:", eps * 255)
    print("dataset:", modelConfig["dataset_name"])
    print("source model:", modelConfig["Source_Model"])
    print("source samples:", len(source_dataset))
    print("target samples:", len(target_dataset))
    print("target classes:", len(targets))

    kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(device)

    ckpt_stem = os.path.splitext(os.path.basename(modelConfig["test_load_weight"]))[0]
    export_root = os.path.join(modelConfig["Generator_save_dir"], f"crafted_{ckpt_stem}")
    source_dir = os.path.join(export_root, "source")
    target_dir = os.path.join(export_root, "target")
    adv_dir = os.path.join(export_root, "adv")
    delta_vis_dir = os.path.join(export_root, "delta_vis")
    delta_tensor_dir = os.path.join(export_root, "delta_tensor")

    for p in [export_root, source_dir, target_dir, adv_dir, delta_vis_dir, delta_tensor_dir]:
        ensure_dir(p)

    craft_config = {
        "dataset_name": modelConfig["dataset_name"],
        "source_model": modelConfig["Source_Model"],
        "checkpoint": modelConfig["test_load_weight"],
        "generator_save_dir": modelConfig["Generator_save_dir"],
        "batch_size": int(modelConfig["batch_size"]),
        "eps": eps,
        "target_select": str(modelConfig.get("target_select", "1")),
        "unknown": bool(modelConfig.get("unknown", False)),
        "set_targets": modelConfig["set_targets"],
        "targets": targets,
        "class_names": class_names,
        "image_size": int(modelConfig["image_size"]),
        "scale_size": int(modelConfig["scale_size"]),
    }

    with open(os.path.join(export_root, "craft_config.json"), "w") as f:
        json.dump(craft_config, f, indent=2)

    metadata_path = os.path.join(export_root, "metadata.jsonl")
    export_count = 0

    with torch.no_grad(), open(metadata_path, "w") as meta_file:
        progress = tqdm(source_loader, dynamic_ncols=True, desc="Crafting adversarial samples")

        for source_images, source_labels, source_indices in progress:
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)

            selected_target_dataset_indices, selected_target_labels = choose_target_indices_for_batch(
                source_labels=source_labels.cpu(),
                target_pools=target_pools,
                allowed_target_labels=targets,
                target_select=str(modelConfig.get("target_select", "1")),
            )

            target_images_list = []
            for idx in selected_target_dataset_indices:
                img, _ = get_dataset_item(target_dataset, idx)
                target_images_list.append(img)

            target_images = torch.stack(target_images_list, dim=0).to(device)
            target_labels_tensor = torch.tensor(selected_target_labels, dtype=torch.long, device=device)

            target_feature = feature_extraction(normalize(target_images)).squeeze()
            output_to_mix = target_feature

            perturbated_imgs = generator(source_images, mix=output_to_mix)
            perturbated_imgs = kernel(perturbated_imgs)

            adv = torch.min(torch.max(perturbated_imgs, source_images - eps), source_images + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
            delta = adv - source_images

            batch_size_now = adv.shape[0]
            for j in range(batch_size_now):
                sample_id = export_count
                stem = f"sample_{sample_id:06d}"

                source_rel = f"source/{stem}.png"
                target_rel = f"target/{stem}.png"
                adv_rel = f"adv/{stem}.png"
                delta_vis_rel = f"delta_vis/{stem}.png"
                delta_tensor_rel = f"delta_tensor/{stem}.pt"

                save_image(source_images[j].cpu(), os.path.join(export_root, source_rel))
                save_image(target_images[j].cpu(), os.path.join(export_root, target_rel))
                save_image(adv[j].cpu(), os.path.join(export_root, adv_rel))
                save_delta_vis(delta[j].cpu(), os.path.join(export_root, delta_vis_rel), magnify=8.0)
                torch.save(delta[j].cpu(), os.path.join(export_root, delta_tensor_rel))

                src_global_idx = int(source_indices[j].item())
                tgt_dataset_idx = int(selected_target_dataset_indices[j])

                src_path = get_dataset_path(source_dataset, src_global_idx)
                tgt_path = get_dataset_path(target_dataset, tgt_dataset_idx)

                record = {
                    "sample_id": sample_id,
                    "source_dataset_index": src_global_idx,
                    "target_dataset_index": tgt_dataset_idx,
                    "source_label": int(source_labels[j].item()),
                    "target_label": int(target_labels_tensor[j].item()),
                    "source_class_name": class_names[int(source_labels[j].item())] if len(class_names) > 0 else None,
                    "target_class_name": class_names[int(target_labels_tensor[j].item())] if len(class_names) > 0 else None,
                    "source_original_path": src_path,
                    "target_original_path": tgt_path,
                    "source_image": source_rel,
                    "target_image": target_rel,
                    "adv_image": adv_rel,
                    "delta_visualization": delta_vis_rel,
                    "delta_tensor": delta_tensor_rel,
                    "linf": float(delta[j].abs().max().item()),
                    "l2": float(delta[j].flatten().norm(p=2).item()),
                }
                meta_file.write(json.dumps(record) + "\n")
                export_count += 1

            progress.set_postfix({
                "saved": export_count,
            })

    craft_summary = {
        "dataset_name": modelConfig["dataset_name"],
        "source_model": modelConfig["Source_Model"],
        "checkpoint": modelConfig["test_load_weight"],
        "num_source_samples": len(source_dataset),
        "num_target_samples": len(target_dataset),
        "num_targets": len(targets),
        "num_exported_samples": export_count,
        "export_root": export_root,
        "metadata_file": metadata_path,
    }

    with open(os.path.join(export_root, "craft_summary.json"), "w") as f:
        json.dump(craft_summary, f, indent=2)

    print("=" * 100)
    print("Crafted adversarial dataset")
    print(f"Export root: {export_root}")
    print(f"Metadata:    {metadata_path}")
    print(f"Samples:     {export_count}")
    print("=" * 100)