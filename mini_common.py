"""Shared helpers for the mini CLIPScope reproduction."""
import torch
import clip
from torchvision.datasets import (
    CIFAR10, CIFAR100, SVHN, EuroSAT, FashionMNIST, MNIST,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    """Return a torchvision dataset (test split, downloaded if missing)."""
    n = name.lower()
    if n == "cifar100":
        return CIFAR100(root, train=False, download=True)
    if n == "cifar10":
        return CIFAR10(root, train=False, download=True)
    if n == "svhn":
        return SVHN(root, split="test", download=True)
    if n == "eurosat":
        return EuroSAT(root, download=True)
    if n == "fashionmnist":
        return FashionMNIST(root, train=False, download=True)
    if n == "mnist":
        return MNIST(root, train=False, download=True)
    raise ValueError(f"Unknown dataset: {name}")


def class_names(ds):
    return [c.replace("_", " ").lower() for c in ds.classes]