import random
import time
import numpy as np
import torch
import clip

from torchvision.datasets import CIFAR10, FashionMNIST
from sklearn.metrics import roc_auc_score, roc_curve

start_time = time.time()

def log(message):
    print(f"[{time.time() - start_time:.1f}s] {message}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")

model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()
log("CLIP model loaded.")

id_data = CIFAR10(root="./data", train=False, download=True)
ood_data = FashionMNIST(root="./data", train=False, download=True)
log("Datasets loaded.")

id_classes = ["airplane", "automobile", "bird", "cat", "deer"]

id_indices = [
    i for i, label in enumerate(id_data.targets)
    if id_data.classes[label] in id_classes
][:100]

ood_indices = list(range(100))

all_nouns = torch.load("nouns.pt", weights_only=False)
all_nouns = [str(w) for w in all_nouns]
log(f"Loaded {len(all_nouns)} nouns.")

rng = random.Random(42)
all_nouns = rng.sample(all_nouns, min(2000, len(all_nouns)))

log(f"Using {len(all_nouns)} nouns for the diagnostic experiment.")

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
log("ID class prompts encoded.")

noun_scores = []
total_batches = (len(all_nouns) + 63) // 64
noun_start = time.time()

for batch_num, start in enumerate(range(0, len(all_nouns), 64), 1):
    batch_nouns = all_nouns[start:start + 64]
    features = encode_texts([f"the {w}" for w in batch_nouns])
    similarities = features @ id_features.T
    noun_scores.extend(similarities.max(dim=1).values.cpu().tolist())

    if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
        log(
            f"Noun ranking: {batch_num}/{total_batches} batches "
            f"({100 * batch_num / total_batches:.1f}%)"
        )

log(f"Noun ranking complete in {time.time() - noun_start:.1f}s.")

order = np.argsort(noun_scores)
far_nouns = [all_nouns[i] for i in order[:50]]
near_nouns = [all_nouns[i] for i in order[-50:]]

ood_prompts = (
    [f"a photo of a {w}" for w in far_nouns]
    + [f"a photo of {w}" for w in near_nouns]
)
ood_features = encode_texts(ood_prompts)
text_features = torch.cat([id_features, ood_features], dim=0)
log("OOD prompts encoded. Starting evaluation.")

def evaluate(order_seed=0):
    rng = np.random.default_rng(order_seed)

    samples = (
        [(i, 0) for i in id_indices]
        + [(i, 1) for i in ood_indices]
    )
    rng.shuffle(samples)

    count = torch.ones(len(id_classes), device=device)
    results = []
    eval_start = time.time()

    for step, (index, label) in enumerate(samples, 1):
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

            results.append((float(confidence.item()), label, pred))
            count[pred] += 1

        if step == 1 or step % 20 == 0 or step == len(samples):
            log(
                f"Seed {order_seed}: {step}/{len(samples)} images "
                f"({100 * step / len(samples):.1f}%)"
            )

    results = np.asarray(results, dtype=np.float64)
    scores = results[:, 0]
    labels = results[:, 1].astype(int)

    auroc = roc_auc_score(labels == 0, scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)

    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = fpr[valid[0]] if len(valid) else np.nan

    assert np.isfinite(scores).all()

    log(f"Seed {order_seed} evaluation took {time.time() - eval_start:.1f}s.")
    return auroc, fpr95

for seed in range(3):
    log(f"Starting run {seed + 1}/3.")
    auroc, fpr95 = evaluate(seed)
    print(
        f"Seed {seed}: "
        f"AUROC={auroc * 100:.2f}%, "
        f"FPR95={fpr95 * 100:.2f}%",
        flush=True
    )

log("All runs completed.")