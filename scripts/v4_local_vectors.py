"""Embed texts with the pinned local encoder. argv: texts.json output.npz. Mean pooling, unit length."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

texts = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
with torch.inference_mode():
    for start in range(0, len(order), 64):
        batch = order[start:start + 64]
        encoded = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=384,
                            return_tensors="pt")
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors[batch] = torch.nn.functional.normalize(pooled, dim=1).numpy()
        if (start // 64) % 25 == 0:
            print(start, "of", len(order), flush=True)
np.savez_compressed(sys.argv[2], texts=np.asarray(texts), vectors=vectors)
print("saved", len(texts))
