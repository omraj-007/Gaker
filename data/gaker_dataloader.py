from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


@dataclass
class SplitConfig:
    seen_ratio: float = 0.7
    split_seed: int = 42
    split_strategy: str = "greedy"   # random / greedy / fixed
    known_classes_override: Optional[List[int]] = None
    samples_per_known_class: Optional[int] = None


@dataclass
class LoaderConfig:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last_train: bool = True


def _extract_image_label(sample):
    if isinstance(sample, (tuple, list)):
        if len(sample) < 2:
            raise ValueError("Dataset sample must contain at least (image, label).")
        return sample[0], int(sample[1])
    raise ValueError("Unsupported dataset sample format.")


def get_labels(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        base_labels = get_labels(dataset.dataset)
        return [int(base_labels[i]) for i in dataset.indices]

    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    labels: List[int] = []
    for idx in range(len(dataset)):
        _, label = _extract_image_label(dataset[idx])
        labels.append(int(label))
    return labels


def filter_dataset_by_classes(dataset: Dataset, allowed_classes: Sequence[int]) -> Subset:
    allowed = set(int(x) for x in allowed_classes)
    labels = get_labels(dataset)
    indices = [idx for idx, label in enumerate(labels) if int(label) in allowed]
    return Subset(dataset, indices)


def build_class_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    labels = get_labels(dataset)
    mapping: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        mapping.setdefault(int(label), []).append(int(idx))
    return mapping


def _resolve_num_known(num_classes: int, seen_ratio: float) -> int:
    num_known = int(round(float(num_classes) * float(seen_ratio)))
    num_known = max(1, min(num_classes - 1, num_known))
    return num_known


@torch.no_grad()
def _compute_class_prototypes(
    dataset: Dataset,
    class_to_indices: Dict[int, List[int]],
    feature_extractor,
    device: str,
    max_proto_samples_per_class: Optional[int] = None,
    split_seed: int = 42,
) -> Dict[int, torch.Tensor]:
    rng = random.Random(split_seed)
    prototypes: Dict[int, torch.Tensor] = {}

    classes = sorted(class_to_indices.keys())
    for cls in tqdm(classes, dynamic_ncols=True, desc="Computing class prototypes"):
        indices = list(class_to_indices[cls])

        if max_proto_samples_per_class is not None and len(indices) > max_proto_samples_per_class:
            indices = rng.sample(indices, max_proto_samples_per_class)

        feats = []
        for idx in indices:
            image, _ = _extract_image_label(dataset[idx])
            image = image.unsqueeze(0).to(device)
            feat = feature_extractor.forward_features(image)
            feats.append(feat.squeeze(0).detach().cpu())

        if len(feats) == 0:
            raise ValueError(f"No samples available to compute prototype for class {cls}.")

        proto = torch.stack(feats, dim=0).mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[int(cls)] = proto

    return prototypes


def _greedy_class_split(
    dataset: Dataset,
    seen_ratio: float,
    split_seed: int,
    feature_extractor,
    device: str,
    max_proto_samples_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    class_to_indices = build_class_to_indices(dataset)
    classes = sorted(class_to_indices.keys())

    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to build known/unknown split.")

    num_known = _resolve_num_known(len(classes), seen_ratio)
    prototypes = _compute_class_prototypes(
        dataset=dataset,
        class_to_indices=class_to_indices,
        feature_extractor=feature_extractor,
        device=device,
        max_proto_samples_per_class=max_proto_samples_per_class,
        split_seed=split_seed,
    )

    rng = random.Random(split_seed)
    selected: List[int] = [int(rng.choice(classes))]
    remaining = [c for c in classes if c not in selected]

    while len(selected) < num_known:
        best_cls = None
        best_score = -float("inf")

        for candidate in remaining:
            candidate_proto = prototypes[int(candidate)]
            dists = []
            for chosen in selected:
                chosen_proto = prototypes[int(chosen)]
                cosine_sim = float(torch.dot(candidate_proto, chosen_proto).item())
                cosine_dist = 1.0 - cosine_sim
                dists.append(cosine_dist)

            score = min(dists) if len(dists) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_cls = int(candidate)

        if best_cls is None:
            raise RuntimeError("Greedy class split failed to choose next class.")

        selected.append(best_cls)
        remaining.remove(best_cls)

    known_classes = sorted(selected)
    unknown_classes = sorted([c for c in classes if c not in known_classes])
    return known_classes, unknown_classes

def subset_k_per_class(dataset: Dataset, allowed_classes: Sequence[int], k: Optional[int], seed: int = 42) -> Subset:
    rng = random.Random(seed)
    labels = get_labels(dataset)

    class_to_indices: Dict[int, List[int]] = {}
    allowed = set(int(x) for x in allowed_classes)

    for idx, label in enumerate(labels):
        label = int(label)
        if label in allowed:
            class_to_indices.setdefault(label, []).append(idx)

    selected_indices: List[int] = []
    for cls in sorted(class_to_indices.keys()):
        idxs = class_to_indices[cls]
        if k is not None and len(idxs) > k:
            idxs = rng.sample(idxs, k)
        selected_indices.extend(idxs)

    return Subset(dataset, selected_indices)

def resolve_known_unknown_classes(
    train_dataset: Dataset,
    split_cfg: SplitConfig,
    feature_extractor=None,
    device: str = "cpu",
    max_proto_samples_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    if split_cfg.known_classes_override is not None:
        class_ids = sorted(set(get_labels(train_dataset)))
        known_classes = sorted(int(x) for x in split_cfg.known_classes_override)
        unknown_classes = sorted([c for c in class_ids if c not in set(known_classes)])
        return known_classes, unknown_classes
    
    class_ids = sorted(set(get_labels(train_dataset)))
    if len(class_ids) < 2:
        raise ValueError("Need at least 2 classes in the training dataset.")

    strategy = split_cfg.split_strategy.lower()
    if strategy == "random":
        rng = random.Random(split_cfg.split_seed)
        shuffled = list(class_ids)
        rng.shuffle(shuffled)
        num_known = _resolve_num_known(len(shuffled), split_cfg.seen_ratio)
        known_classes = sorted(shuffled[:num_known])
        unknown_classes = sorted(shuffled[num_known:])
        return known_classes, unknown_classes

    if strategy == "greedy":
        if feature_extractor is None:
            raise ValueError("feature_extractor is required for split_strategy='greedy'.")
        return _greedy_class_split(
            dataset=train_dataset,
            seen_ratio=split_cfg.seen_ratio,
            split_seed=split_cfg.split_seed,
            feature_extractor=feature_extractor,
            device=device,
            max_proto_samples_per_class=max_proto_samples_per_class,
        )

    raise ValueError(f"Unsupported split_strategy: {split_cfg.split_strategy}")


@torch.no_grad()
def filter_target_pool_by_classifier_confidence(
    dataset: Dataset,
    class_to_indices: Dict[int, List[int]],
    classifier_for_quality_filter,
    device: str,
    topk_per_class: int,
) -> Dict[int, List[int]]:
    filtered: Dict[int, List[int]] = {}

    for cls, indices in tqdm(class_to_indices.items(), dynamic_ncols=True, desc="Filtering target pools"):
        scores: List[Tuple[float, int]] = []

        for idx in indices:
            image, label = _extract_image_label(dataset[idx])
            image = image.unsqueeze(0).to(device)
            label_tensor = torch.tensor([int(label)], device=device)

            logits = classifier_for_quality_filter.forward_logits(image)
            prob = F.softmax(logits, dim=1)[0, label_tensor.item()].item()
            scores.append((float(prob), int(idx)))

        scores.sort(key=lambda x: x[0], reverse=True)
        keep = [idx for _, idx in scores[: min(topk_per_class, len(scores))]]
        filtered[int(cls)] = keep

    return filtered


class GakerPairDataset(Dataset):
    """
    Produces:
        (source_image, source_label, target_image, target_label)

    target_mode:
        - "known"
        - "unknown"
        - "all"

    Important:
        - source samples come from base_dataset
        - target samples come from target_base_dataset if provided, else from base_dataset
        - target_pool_by_class can be supplied to lock the target set explicitly
    """

    def __init__(
        self,
        base_dataset: Dataset,
        known_classes: Sequence[int],
        unknown_classes: Sequence[int],
        target_mode: str = "known",
        target_pool_by_class: Optional[Dict[int, List[int]]] = None,
        target_base_dataset: Optional[Dataset] = None,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.target_base_dataset = target_base_dataset if target_base_dataset is not None else base_dataset
        self.known_classes = sorted(int(x) for x in known_classes)
        self.unknown_classes = sorted(int(x) for x in unknown_classes)
        self.target_mode = target_mode.lower()
        self.seed = int(seed)

        if self.target_mode not in {"known", "unknown", "all"}:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        if target_pool_by_class is None:
            target_labels = get_labels(self.target_base_dataset)
            target_pool_by_class = {}
            allowed_classes = self._allowed_target_classes()
            allowed_set = set(allowed_classes)
            for idx, label in enumerate(target_labels):
                if int(label) in allowed_set:
                    target_pool_by_class.setdefault(int(label), []).append(int(idx))

        self.target_pool_by_class = {
            int(k): [int(x) for x in v] for k, v in target_pool_by_class.items()
        }

    def _allowed_target_classes(self) -> List[int]:
        if self.target_mode == "known":
            return list(self.known_classes)
        if self.target_mode == "unknown":
            return list(self.unknown_classes)
        return sorted(list(set(self.known_classes) | set(self.unknown_classes)))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        source_image, source_label = _extract_image_label(self.base_dataset[idx])

        rng = random.Random(self.seed + int(idx))
        candidate_classes = [
            c for c in self._allowed_target_classes()
            if c != int(source_label) and c in self.target_pool_by_class and len(self.target_pool_by_class[c]) > 0
        ]

        if len(candidate_classes) == 0:
            raise RuntimeError(
                f"No valid target class available for source label {source_label} under target_mode={self.target_mode}."
            )

        target_label = int(rng.choice(candidate_classes))
        target_idx = int(rng.choice(self.target_pool_by_class[target_label]))
        target_image, resolved_target_label = _extract_image_label(self.target_base_dataset[target_idx])

        return source_image, int(source_label), target_image, int(resolved_target_label)


def build_gaker_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    split_cfg: SplitConfig,
    loader_cfg: LoaderConfig,
    feature_extractor=None,
    device: str = "cpu",
    max_proto_samples_per_class: Optional[int] = None,
    topk_targets_per_known_class: Optional[int] = None,
    classifier_for_quality_filter=None,
) -> Dict[str, object]:
    known_classes, unknown_classes = resolve_known_unknown_classes(
        train_dataset=train_dataset,
        split_cfg=split_cfg,
        feature_extractor=feature_extractor,
        device=device,
        max_proto_samples_per_class=max_proto_samples_per_class,
    )

    if split_cfg.samples_per_known_class is not None:
        known_train_dataset = subset_k_per_class(
            dataset=train_dataset,
            allowed_classes=known_classes,
            k=split_cfg.samples_per_known_class,
            seed=split_cfg.split_seed,
        )
    else:
        known_train_dataset = filter_dataset_by_classes(train_dataset, known_classes)

    unknown_train_dataset = filter_dataset_by_classes(train_dataset, unknown_classes)
    
    known_test_dataset = filter_dataset_by_classes(test_dataset, known_classes)
    unknown_test_dataset = filter_dataset_by_classes(test_dataset, unknown_classes)

    known_train_pool = build_class_to_indices(known_train_dataset)
    if topk_targets_per_known_class is not None and classifier_for_quality_filter is not None:
        known_train_pool = filter_target_pool_by_classifier_confidence(
            dataset=known_train_dataset,
            class_to_indices=known_train_pool,
            classifier_for_quality_filter=classifier_for_quality_filter,
            device=device,
            topk_per_class=int(topk_targets_per_known_class),
        )

    known_test_pool = build_class_to_indices(known_test_dataset)
    unknown_test_pool = build_class_to_indices(unknown_test_dataset)

    train_pair_dataset = GakerPairDataset(
        base_dataset=known_train_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        target_mode="known",
        target_pool_by_class=known_train_pool,
        target_base_dataset=known_train_dataset,
        seed=split_cfg.split_seed,
    )

    eval_known_dataset = GakerPairDataset(
        base_dataset=known_test_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        target_mode="known",
        target_pool_by_class=known_test_pool,
        target_base_dataset=known_test_dataset,
        seed=split_cfg.split_seed + 1000,
    )

    eval_unknown_dataset = GakerPairDataset(
        base_dataset=known_test_dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        target_mode="unknown",
        target_pool_by_class=unknown_test_pool,
        target_base_dataset=unknown_test_dataset,
        seed=split_cfg.split_seed + 2000,
    )

    train_loader = DataLoader(
        train_pair_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle_train,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last_train,
    )

    eval_known_loader = DataLoader(
        eval_known_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
    )

    eval_unknown_loader = DataLoader(
        eval_unknown_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
    )

    bundle: Dict[str, object] = {
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "known_train_dataset": known_train_dataset,
        "unknown_train_dataset": unknown_train_dataset,
        "known_test_dataset": known_test_dataset,
        "unknown_test_dataset": unknown_test_dataset,
        "train_pair_dataset": train_pair_dataset,
        "eval_known_dataset": eval_known_dataset,
        "eval_unknown_dataset": eval_unknown_dataset,
        "train_loader": train_loader,
        "eval_known_loader": eval_known_loader,
        "eval_unknown_loader": eval_unknown_loader,
        "known_train_pool_sizes": {int(k): len(v) for k, v in known_train_pool.items()},
        "known_test_pool_sizes": {int(k): len(v) for k, v in known_test_pool.items()},
        "unknown_test_pool_sizes": {int(k): len(v) for k, v in unknown_test_pool.items()},
    }

    return bundle
