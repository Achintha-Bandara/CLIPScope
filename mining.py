import nltk
from nltk.corpus import wordnet as wn
import clip
import torch
import numpy as np

nouns = torch.load('nouns.pt')
nouns = np.array(nouns)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
with open('imagenet_class_clean.npy', 'rb') as f:
    imagenet_cls = np.load(f)

text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_cls]).to(device)
percentile = 50
with torch.no_grad():
    text_features = model.encode_text(text).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for word in nouns:
        text = torch.cat([clip.tokenize(f"the "+word)]).to(device)
        sample_features = model.encode_text(text).float()
        sample_features /= sample_features.norm(dim=-1, keepdim=True)
        similarity = -(sample_features @ text_features.T)[0]
        value, index = torch.kthvalue(similarity, percentile)
        scores.append(-value.cpu().numpy())

scores = np.asarray(scores, dtype=np.float32)
sorted_indices = np.argsort(scores)

# Modified to save actual noun strings, not integer indices
sorted_nouns = [str(nouns[i]) for i in sorted_indices]
torch.save(sorted_nouns, "sorted_nouns.pt")

print("Total nouns:", len(sorted_nouns))
print("Lowest-ranked nouns:", sorted_nouns[:10])
print("Highest-ranked nouns:", sorted_nouns[-10:])
