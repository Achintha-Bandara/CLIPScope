import numpy as np
import torch
import clip

from torchvision.datasets import CIFAR10, FashionMNIST
from sklearn.metrics import roc_auc_score, roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

# Set download=False if the datasets are already downloaded
id_data = CIFAR10(
    root="./data", train=False, download=True
)
ood_data = FashionMNIST(
    root="./data", train=False, download=True
)

id_classes = ["airplane", "automobile", "bird", "cat", "deer"]

id_indices = [
    i for i, label in enumerate(id_data.targets)
    if id_data.classes[label] in id_classes
][:100]

ood_indices = list(range(100))

# This is an adapted experiment, not the paper's exact
# ImageNet-based label-mining protocol.
all_nouns = torch.load("nouns.pt", weights_only=False)
all_nouns = [str(w) for w in all_nouns]

def encode_texts(prompts, batch_size=64):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = clip.tokenize(
                prompts[start:start + batch_size]
            ).to(device)
            features = model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
            chunks.append(features)
    return torch.cat(chunks, dim=0)

id_features = encode_texts(
    [f"a photo of a {c}" for c in id_classes]
)

noun_scores = []

for start in range(0, len(all_nouns), 64):
    batch_nouns = all_nouns[start:start + 64]
    features = encode_texts([f"the {w}" for w in batch_nouns])
    similarities = features @ id_features.T
    noun_scores.extend(similarities.max(dim=1).values.cpu().tolist())

order = np.argsort(noun_scores)
far_nouns = [all_nouns[i] for i in order[:50]]
near_nouns = [all_nouns[i] for i in order[-50:]]

ood_prompts = (
    [f"a photo of a {w}" for w in far_nouns]
    + [f"a photo of {w}" for w in near_nouns]
)
ood_features = encode_texts(ood_prompts)

text_features = torch.cat(
    [id_features, ood_features], dim=0
)

def evaluate(order_seed=0):
    rng = np.random.default_rng(order_seed)

    samples = (
        [(i, 0) for i in id_indices]
        + [(i, 1) for i in ood_indices]
    )
    rng.shuffle(samples)

    count = torch.ones(len(id_classes), device=device)
    results = []

    for index, label in samples:
        if label == 0:
            image, _ = id_data[index]
        else:
            image, _ = ood_data[index]

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(
                dim=-1, keepdim=True
            )

            similarity = image_features @ text_features.T

            probs = (100 * similarity[:, :len(id_classes)]).softmax(
                dim=-1
            )[0]

            pred = int(probs.argmax().item())

            p = (100 * similarity).softmax(dim=-1)[
                0, :len(id_classes)
            ].sum()

            class_frequency = count / count.sum()
            confidence = probs[pred] * p / class_frequency[pred]

            results.append(
                (float(confidence.item()), label, pred)
            )
            count[pred] += 1

    results = np.asarray(results, dtype=np.float64)
    scores = results[:, 0]
    labels = results[:, 1].astype(int)

    # Higher scores are intended to indicate ID samples.
    auroc = roc_auc_score(labels == 0, scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)

    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = fpr[valid[0]] if len(valid) else np.nan

    assert np.isfinite(scores).all()
    return auroc, fpr95

for seed in range(3):
    auroc, fpr95 = evaluate(seed)
    print(
        f"Seed {seed}: "
        f"AUROC={auroc * 100:.2f}%, "
        f"FPR95={fpr95 * 100:.2f}%"
    )