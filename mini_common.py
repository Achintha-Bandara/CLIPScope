"""Shared helpers for the mini CLIPScope reproduction."""
from pathlib import Path

import numpy as np
import torch
import clip
from torchvision.datasets import (
    CIFAR10, CIFAR100, SVHN, EuroSAT, FashionMNIST, MNIST, KMNIST,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent


def load_clip(name="ViT-B/16"):
    model, preprocess = clip.load(name, DEVICE)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_texts(model, prompts, batch_size=256):
    """L2-normalised CLIP text features, computed in batches (CPU tensor out)."""
    feats = []
    for s in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[s:s + batch_size], truncate=True).to(DEVICE)
        f = model.encode_text(tokens).float()
        f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats)


def get_dataset(name, root="data"):
    """Return a torchvision-style dataset (test split, downloaded if missing)."""
    n = name.lower()
    if n == "cifar100":
        return CIFAR100(root, train=False, download=True)
    if n == "cifar10":
        return CIFAR10(root, train=False, download=True)
    if n == "svhn":
        return SVHN(root, split="test", download=True)
    if n == "eurosat":      # NOTE: overlaps semantically with CIFAR-100 (forest, sea, river, road)
        return EuroSAT(root, download=True)
    if n == "fashionmnist":
        return FashionMNIST(root, train=False, download=True)
    if n == "mnist":
        return MNIST(root, train=False, download=True)
    if n == "kmnist":
        return KMNIST(root, train=False, download=True)
    # if n == "dtd":          # Textures, as in the paper (~600 MB)
    #     from torchvision.datasets import DTD
    #     return DTD(root, split="test", download=True)
    # if n == "imagenetv2":   # ID set with the paper's 1000 ImageNet classes (~1.2 GB, 10k images)
    #     from imagenetv2_pytorch import ImageNetV2Dataset
    #     return ImageNetV2Dataset("matched-frequency")
    raise ValueError(f"Unknown dataset: {name}")


def class_names(x):
    """x: dataset name or dataset object. ImageNetV2 uses imagenet_class_clean.npy."""
    if isinstance(x, str):
        if x.lower() == "imagenetv2":
            return [str(c) for c in np.load(ROOT / "imagenet_class_clean.npy").tolist()]
        x = get_dataset(x)
    return [c.replace("_", " ").lower() for c in x.classes]