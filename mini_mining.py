from pathlib import Path

import numpy as np
import torch
import clip


ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]
MAX_CANDIDATES = 2000
M = 50
PERCENTILE = 0.05
TEXT_BATCH_SIZE = 64

# Used when the optional nouns.pt generated from WordNet is unavailable.
FALLBACK_NOUNS = [
    "apple", "backpack", "balloon", "banana", "bed", "bench", "bicycle",
    "blanket", "boat", "book", "bottle", "box", "bread", "bridge",
    "bucket", "bus", "butterfly", "cabinet", "camera", "candle", "carrot",
    "castle", "chair", "clock", "cloud", "coat", "computer", "couch",
    "desk", "door", "drum", "elephant", "engine", "factory", "feather",
    "fence", "flower", "fork", "forest", "fridge", "guitar", "hammer",
    "hat", "helicopter", "horse", "house", "island", "jacket", "keyboard",
    "knife", "lamp", "laptop", "leaf", "lion", "mountain", "mouse",
    "mushroom", "ocean", "orange", "pancake", "parachute", "pencil",
    "penguin", "phone", "piano", "pizza", "plane", "plate", "pumpkin",
    "rabbit", "rainbow", "river", "robot", "rocket", "rose", "scooter",
    "ship", "shoe", "skateboard", "snake", "sofa", "spoon", "square",
    "star", "strawberry", "street", "suitcase", "sunflower", "table",
    "telephone", "television", "tent", "tiger", "train", "truck", "umbrella",
    "vase", "violin", "wallet", "watermelon", "wheel", "window", "zebra",
]


def load_candidates():
    path = ROOT / "nouns.pt"
    if path.exists():
        try:
            candidates = torch.load(path, weights_only=False)
        except TypeError:
            candidates = torch.load(path)
        if not isinstance(candidates, (list, tuple, set)):
            raise TypeError(f"{path} must contain a list-like noun collection.")
        source = str(path)
    else:
        candidates = FALLBACK_NOUNS
        source = "built-in fallback vocabulary (nouns.pt not found)"

    candidates = {
        str(word).replace("_", " ").strip().lower()
        for word in candidates
        if str(word).strip()
    }
    candidates.difference_update(ID_CLASSES)
    candidates = sorted(candidates)
    return candidates[:MAX_CANDIDATES], source


def encode_texts(model, prompts, batch_size=64):
    if not prompts:
        raise ValueError("Cannot encode an empty prompt collection.")

    features = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = clip.tokenize(prompts[start:start + batch_size]).to(DEVICE)
            batch = model.encode_text(tokens).float()
            norms = batch.norm(dim=-1, keepdim=True)
            if torch.any(norms == 0):
                raise ValueError("CLIP returned a zero-norm text embedding.")
            batch /= norms
            features.append(batch)
    return torch.cat(features)


def percentile_index(size, percentile):
    if size < 1 or not 0 < percentile <= 1:
        raise ValueError("size must be positive and percentile must be in (0, 1].")
    return max(1, min(size, int(np.ceil(size * percentile))))


def main():
    candidates, source = load_candidates()
    if len(candidates) < 2 * M:
        raise ValueError(
            f"Need at least {2 * M} candidate nouns, but found {len(candidates)}."
        )

    model, _ = clip.load("ViT-B/16", device=DEVICE)
    model.eval()

    id_features = encode_texts(
        model, [f"a photo of a {label}" for label in ID_CLASSES]
    )
    candidate_features = encode_texts(
        model,
        [f"a photo of a {word}" for word in candidates],
        batch_size=TEXT_BATCH_SIZE,
    )

    negative_cosine = -(candidate_features @ id_features.T)
    kth = percentile_index(len(ID_CLASSES), PERCENTILE)
    distances = torch.kthvalue(negative_cosine, kth, dim=1).values
    order = torch.argsort(distances).cpu().tolist()

    ranked_nouns = [candidates[index] for index in order]
    ranked_path = ROOT / "mini_sorted_nouns.pt"
    selected_path = ROOT / "mini_ood_labels.pt"
    torch.save(ranked_nouns, ranked_path)

    nearest = ranked_nouns[:M]
    farthest = ranked_nouns[-M:][::-1]
    selected = farthest + nearest
    torch.save(selected, selected_path)

    print(f"Candidate source: {source}")
    print(f"Candidates: {len(candidates)}")
    print(f"Percentile: {PERCENTILE:.0%} ({kth}th order statistic)")
    print(f"Selected OOD labels: {len(selected)}")
    print(f"Saved ranked nouns to {ranked_path}")
    print(f"Saved selected labels to {selected_path}")
    print(f"Farthest labels: {farthest}")
    print(f"Nearest labels: {nearest}")


if __name__ == "__main__":
    main()
