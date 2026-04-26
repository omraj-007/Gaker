import os
import json
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import torchvision
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from generator.Generator import Generator
import torch.nn as nn
import torch.nn.functional as F
from utils.gaussian_smoothing import get_gaussian_kernel


def get_device_count() -> int:
    return torch.cuda.device_count()


def normalize(t: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = t.clone()
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


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


def seed_torch(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.03, 0.33),
            ratio=(0.3, 3.3),
            value=0,
            inplace=False,
        ),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    return train_transform, target_transform


def get_dataset_labels(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))
    return labels


def build_cifar_dataset(dataset_name: str, root: str, train: bool, transform):
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    if dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    raise ValueError(f"Unsupported CIFAR dataset: {dataset_name}")


def build_imagefolder_dataset(dataset_name: str, root: str, transform):
    if dataset_name == "tinyimagenet":
        resolved_root = resolve_imagefolder_root(root, ["train"])
        return torchvision.datasets.ImageFolder(resolved_root, transform=transform)

    if dataset_name == "imagenet":
        # if root/train exists, use it; otherwise assume root itself is class-folder formatted
        resolved_root = resolve_imagefolder_root(root, ["train"])
        return torchvision.datasets.ImageFolder(resolved_root, transform=transform)

    raise ValueError(f"Unsupported ImageFolder dataset: {dataset_name}")


def build_train_and_target_datasets(modelConfig: Dict, train_transform, target_transform):
    dataset_name = modelConfig["dataset_name"].lower()

    if dataset_name in {"cifar10", "cifar100"}:
        train_set = build_cifar_dataset(
            dataset_name=dataset_name,
            root=modelConfig["datasets_root"],
            train=True,
            transform=train_transform,
        )
        target_base_set = build_cifar_dataset(
            dataset_name=dataset_name,
            root=modelConfig["datasets_root"],
            train=True,
            transform=target_transform,
        )
    elif dataset_name in {"tinyimagenet", "imagenet"}:
        train_set = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["train_dir"],
            transform=train_transform,
        )
        target_base_set = build_imagefolder_dataset(
            dataset_name=dataset_name,
            root=modelConfig["target_dir"],
            transform=target_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    class_names = list(getattr(train_set, "classes", [str(i) for i in range(modelConfig["num_classes"])]))

    if hasattr(train_set, "class_to_idx"):
        class_to_idx = {str(k): int(v) for k, v in train_set.class_to_idx.items()}
    else:
        class_to_idx = {str(name): int(i) for i, name in enumerate(class_names)}

    num_classes = len(class_names)

    if dataset_name == "imagenet":
        split_seed = int(modelConfig.get("split_seed", 42))
        num_known_classes = int(modelConfig.get("num_known_classes", 200))
        samples_per_known_class = int(modelConfig.get("samples_per_known_class", 325))

        rng = random.Random(split_seed)
        all_classes = list(range(num_classes))
        rng.shuffle(all_classes)
        targets = sorted(all_classes[:num_known_classes])

        train_set = subset_k_per_class_imagefolder(
            dataset=train_set,
            allowed_classes=set(targets),
            k=samples_per_known_class,
            seed=split_seed,
        )

        target_set = subset_k_per_class_imagefolder(
            dataset=target_base_set,
            allowed_classes=set(targets),
            k=samples_per_known_class,
            seed=split_seed + 1,
        )

    else:
        targets = resolve_target_list(
            set_targets=modelConfig["set_targets"],
            num_classes=num_classes,
        )

        target_labels = get_dataset_labels(target_base_set)
        target_indices = [i for i, label in enumerate(target_labels) if int(label) in targets]
        target_set = Subset(target_base_set, target_indices)

    return train_set, target_set, class_names, class_to_idx, targets


def resolve_target_list(set_targets: str, num_classes: int) -> List[int]:
    if set_targets == "targets_200":
        if num_classes != 1000:
            raise ValueError(
                "set_targets='targets_200' is only valid for the original ImageNet-1K setup. "
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
        f"Unsupported set_targets: {set_targets}. "
        f"Use 'all_classes' or 'targets_200'."
    )


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


def save_training_metadata(
    modelConfig: Dict,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    targets: List[int],
    train_len: int,
    target_len: int,
) -> None:
    save_dir = modelConfig["Generator_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    dataset_info = {
        "dataset_name": modelConfig["dataset_name"],
        "datasets_root": modelConfig["datasets_root"],
        "dataset_root": modelConfig.get("dataset_root", ""),
        "train_dir": modelConfig.get("train_dir", ""),
        "target_dir": modelConfig.get("target_dir", ""),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "targets": targets,
        "num_train_samples": int(train_len),
        "num_target_samples": int(target_len),
        "source_model": modelConfig["Source_Model"],
        "image_size": int(modelConfig["image_size"]),
        "scale_size": int(modelConfig["scale_size"]),
        "eps": float(modelConfig["eps"]),
        "batch_size": int(modelConfig["batch_size"]),
        "lr": float(modelConfig["lr"]),
        "num_res_blocks": int(modelConfig["num_res_blocks"]),
        "channel": int(modelConfig["channel"]),
        "channel_mult": list(modelConfig["channel_mult"]),
        "seed": int(modelConfig.get("seed", 42)),
    }

    with open(os.path.join(save_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    with open(os.path.join(save_dir, "targets.json"), "w") as f:
        json.dump({"targets": targets}, f, indent=2)


def subset_k_per_class_imagefolder(dataset, allowed_classes, k=None, seed=42):
    rng = random.Random(seed)
    class_to_indices = {}

    for idx, (_, label) in enumerate(dataset.samples):
        label = int(label)
        if label in allowed_classes:
            class_to_indices.setdefault(label, []).append(idx)

    selected_indices = []
    for cls in sorted(class_to_indices.keys()):
        idxs = class_to_indices[cls]
        if k is not None and len(idxs) > k:
            rng.shuffle(idxs)
            idxs = idxs[:k]
        selected_indices.extend(idxs)

    return Subset(dataset, selected_indices)


def train(modelConfig: Dict):
    time_start = time.time()
    seed_torch(int(modelConfig.get("seed", 42)))

    device_count = get_device_count()
    print("gpu_num", device_count)

    requested_device = modelConfig["device"]
    if torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        device = torch.device("cpu")

    train_transform, target_transform = get_transforms(modelConfig)

    train_set, target_set, class_names, class_to_idx, targets = build_train_and_target_datasets(
        modelConfig=modelConfig,
        train_transform=train_transform,
        target_transform=target_transform,
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

    num_workers = int(modelConfig.get("num_workers", 12))
    batch_size = int(modelConfig["batch_size"])

    dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dataloader_target = DataLoader(
        target_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    learning_rate = float(modelConfig["lr"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=learning_rate,
        weight_decay=5e-5,
    )

    save_dir = modelConfig["Generator_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    save_training_metadata(
        modelConfig=modelConfig,
        class_names=class_names,
        class_to_idx=class_to_idx,
        targets=targets,
        train_len=len(train_set),
        target_len=len(target_set),
    )

    eps = float(modelConfig["eps"])
    noise_loss_weight = float(modelConfig.get("noise_loss_weight", 0.5))
    print("learning rate", learning_rate)
    print("eps:", eps * 255)
    print("dataset:", modelConfig["dataset_name"])
    print("source model:", modelConfig["Source_Model"])
    print("train samples:", len(train_set))
    print("target samples:", len(target_set))
    print("num targets:", len(targets))

    kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(device)

    history = []

    for e in range(int(modelConfig["epoch"])):
        iteration = 0
        epoch_losses = []

        target_iter = iter(dataloader_target)

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()

                try:
                    imgs_target, labels_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(dataloader_target)
                    imgs_target, labels_target = next(target_iter)

                images = images.to(device)
                labels = labels.to(device)
                imgs_target = imgs_target.to(device)
                labels_target = labels_target.to(device)

                if imgs_target.shape[0] != batch_size or images.shape[0] != batch_size:
                    continue

                with torch.no_grad():
                    target_fea = feature_extraction(normalize(imgs_target.clone().detach())).squeeze()

                output_to_mix = target_fea
                target_feature = target_fea

                mask = torch.ne(labels, labels_target).float().to(device)

                perturbated_imgs = kernel(generator(images, mix=output_to_mix))
                adv = torch.min(torch.max(perturbated_imgs, images - eps), images + eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                adv_feature = feature_extraction(normalize(adv)).squeeze()
                loss = 1 - torch.cosine_similarity(adv_feature, target_feature, dim=1)
                loss = mask * loss

                noise = adv - images
                noise_feature = feature_extraction(normalize(noise)).squeeze()
                loss_noise = 1 - torch.cosine_similarity(noise_feature, target_feature, dim=1)
                loss_noise = mask * loss_noise * noise_loss_weight

                loss = loss + loss_noise
                loss = loss.sum() / images.shape[0]

                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                epoch_losses.append(loss_value)

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss_value,
                })

                if iteration % 1000 == 0:
                    with open(os.path.join(save_dir, "loss.txt"), "a") as f:
                        f.write(f"epoch {e}: iter {iteration}: loss {loss_value}\n")

                iteration += 1

        mean_epoch_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else None
        history.append({
            "epoch": int(e),
            "mean_loss": mean_epoch_loss,
            "num_updates": int(len(epoch_losses)),
        })

        with open(os.path.join(save_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        torch.cuda.empty_cache()

        # Keep checkpoint format compatible with original craftadv.py
        checkpoint_name = f"ckpt_{e}_{modelConfig['Source_Model']}_.pt"
        torch.save(generator.state_dict(), os.path.join(save_dir, checkpoint_name))

    final_summary = {
        "dataset_name": modelConfig["dataset_name"],
        "source_model": modelConfig["Source_Model"],
        "epochs": int(modelConfig["epoch"]),
        "last_epoch_checkpoint": f"ckpt_{int(modelConfig['epoch']) - 1}_{modelConfig['Source_Model']}_.pt",
        "history": history,
        "save_dir": save_dir,
        "elapsed_seconds": float(time.time() - time_start),
    }

    with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    time_end = time.time()
    print("time cost:", time_end - time_start, "s")