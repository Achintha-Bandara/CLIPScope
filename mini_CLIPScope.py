from pathlib import Path
import random

import numpy as np
import torch
import clip
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.datasets import CIFAR10, FashionMNIST


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]
ID_LIMIT_PER_CLASS = 20
OOD_LIMIT = 100
SEEDS = (0, 1, 2)


def encode_texts(model, prompts, batch_size=64):
    features = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = clip.tokenize(prompts[start:start + batch_size]).to(DEVICE)
            batch = model.encode_text(tokens).float()
            batch /= batch.norm(dim=-1, keepdim=True)
            features.append(batch)
    return torch.cat(features)


def load_ood_labels():
    path = ROOT / "mini_ood_labels.pt"
    if not path.exists():
        raise FileNotFoundError(
            "mini_ood_labels.pt is missing. Run mini_mining.py first."
        )
    labels = torch.load(path, weights_only=False)
    return [str(label) for label in labels]


def load_samples():
    id_data = CIFAR10(root=str(DATA_ROOT), train=False, download=False)
    ood_data = FashionMNIST(root=str(DATA_ROOT), train=False, download=False)

    class_to_index = {name: index for index, name in enumerate(id_data.classes)}
    selected_indices = []
    for class_name in ID_CLASSES:
        class_index = class_to_index[class_name]
        matches = [
            index for index, target in enumerate(id_data.targets)
            if target == class_index
        ]
        selected_indices.extend(matches[:ID_LIMIT_PER_CLASS])

    return id_data, ood_data, selected_indices, list(range(OOD_LIMIT))


def make_text_features(model, ood_labels):
    id_features = encode_texts(
        model, [f"a photo of a {label}" for label in ID_CLASSES]
    )
    ood_features = encode_texts(
        model, [f"a photo of a {label}" for label in ood_labels]
    )
    return torch.cat([id_features, ood_features])


def evaluate(model, preprocess, text_features, id_data,
             ood_data, id_indices, ood_indices, seed):
    samples = [(index, 0) for index in id_indices]
    samples.extend((index, 1) for index in ood_indices)
    random.Random(seed).shuffle(samples)

    counts = torch.ones(len(ID_CLASSES), device=DEVICE)
    results = []
    for index, label in samples:
        image = id_data[index][0] if label == 0 else ood_data[index][0]
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T

            id_probabilities = (100 * similarity[:, :len(ID_CLASSES)]).softmax(
                dim=-1
            )[0]
            predicted_class = int(id_probabilities.argmax().item())
            p1 = id_probabilities[predicted_class]

            all_probabilities = (100 * similarity).softmax(dim=-1)[0]
            p2 = all_probabilities[:len(ID_CLASSES)].sum()

            p0 = counts[predicted_class] / counts.sum()
            score = p1 * p2 / p0
            results.append((float(score.item()), label))

            # The current sample must not influence its own p0.
            counts[predicted_class] += 1

    values = np.asarray(results, dtype=np.float64)
    scores = values[:, 0]
    labels = values[:, 1].astype(int)
    auroc = roc_auc_score(labels == 0, scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = fpr[valid[0]] if len(valid) else np.nan
    return auroc, fpr95


def main():
    ood_labels = load_ood_labels()
    id_data, ood_data, id_indices, ood_indices = load_samples()

    model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    model.eval()
    text_features = make_text_features(model, ood_labels)

    print(f"Device: {DEVICE}")
    print(f"ID samples: {len(id_indices)}")
    print(f"OOD samples: {len(ood_indices)}")
    print(f"OOD labels: {len(ood_labels)}")

    aurocs = []
    fprs = []
    for seed in SEEDS:
        auroc, fpr95 = evaluate(
            model, preprocess, text_features, id_data,
            ood_data, id_indices, ood_indices, seed
        )
        aurocs.append(auroc)
        fprs.append(fpr95)
        print(
            f"seed={seed} AUROC={auroc * 100:.2f}% "
            f"FPR95={fpr95 * 100:.2f}%"
        )

    print(f"mean AUROC={np.mean(aurocs) * 100:.2f}%")
    print(f"mean FPR95={np.nanmean(fprs) * 100:.2f}%")


if __name__ == "__main__":
    main()
