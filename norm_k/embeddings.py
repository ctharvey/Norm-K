"""Deterministic offline control and opt-in, capped production embedding adapter.

The hashed encoder is NOT a learned embedding model; it is only for validating
pipeline wiring, caching, numerical invariants, and report generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class Encoder(Protocol):
    name: str
    dimensions: int

    def encode(self, texts: Sequence[str]) -> list[NDArray[np.float64]]: ...


class HashingEncoder:
    """Signed lexical feature hash; use for smoke tests only."""

    def __init__(self, dimensions: int = 256):
        if dimensions < 2:
            raise ValueError("HashingEncoder dimensions must be >= 2")
        self.dimensions = dimensions
        self.name = f"hashing-control-v1-d{dimensions}"

    def encode(self, texts: Sequence[str]) -> list[NDArray[np.float64]]:
        result = []
        for text in texts:
            words = re.findall(r"[a-z0-9]+", text.lower())
            v = np.zeros(self.dimensions, dtype=np.float64)
            # Bigrams deliberately introduce local interaction in a lexical control.
            for token in words + ["__bigram__" + a + "|" + b for a, b in zip(words, words[1:])]:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = -1.0 if digest[8] & 1 else 1.0
                v[idx] += sign
            norm = float(np.linalg.norm(v))
            if norm:
                v /= norm
            result.append(v)
        return result


class OpenAIEncoder:
    """Opt-in direct embeddings API client; no requests without a configured key.

    The caller controls invocation and a separate cap on *new* cache misses.
    The API key is never included in repr, cache files, or error output.
    """

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.name = model
        self.dimensions = dimensions
        self._api_key = os.environ.get("OPENAI_API_KEY")

    def encode(self, texts: Sequence[str]) -> list[NDArray[np.float64]]:
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured; use --encoder hash for offline smoke tests")
        request_body = json.dumps({"model": self.name, "input": list(texts),
                                   "dimensions": self.dimensions}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings", data=request_body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=35) as response:
                    payload = json.load(response)
                indexed = sorted(payload["data"], key=lambda record: record["index"])
                if len(indexed) != len(texts):
                    raise RuntimeError("API returned a different number of embeddings")
                vectors = [np.asarray(item["embedding"], dtype=np.float64) for item in indexed]
                if any(v.shape != (self.dimensions,) or not np.isfinite(v).all() for v in vectors):
                    raise RuntimeError("Embedding API returned invalid dimensions or numbers")
                return vectors
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Embedding API HTTP {exc.code}") from None
                time.sleep(2 ** attempt)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Embedding API network failure: {exc.reason}") from None
        raise RuntimeError("Embedding API retries exhausted")


class EmbeddingCache:
    """SQLite cache namespaced by exact encoder identity and text contents."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, encoder TEXT NOT NULL, dimensions INTEGER NOT NULL, "
            "data BLOB NOT NULL)"
        )

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _key(encoder: Encoder, text: str) -> str:
        key = json.dumps([encoder.name, encoder.dimensions, text], ensure_ascii=False)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def lookup(self, encoder: Encoder, text: str) -> NDArray[np.float64] | None:
        row = self.connection.execute(
            "SELECT data, dimensions FROM embeddings WHERE cache_key = ?",
            (self._key(encoder, text),),
        ).fetchone()
        if row is None:
            return None
        binary, dimensions = row
        if dimensions != encoder.dimensions:
            raise ValueError("Cached vector has wrong dimensions")
        v = np.frombuffer(binary, dtype="<f8").copy()
        if v.size != encoder.dimensions or not np.isfinite(v).all():
            raise ValueError("Corrupt cached vector")
        return v

    def embed(self, encoder: Encoder, texts: Sequence[str],
              max_new: int = 0, batch_size: int = 64) -> list[NDArray[np.float64]]:
        """Resolve vectors; cap non-hash misses to avoid surprise paid API use.

        Preflight the cap before any remote request so partial paid runs do not
        occur merely because the requested batch was larger than expected.
        """
        if batch_size < 1 or max_new < 0:
            raise ValueError("batch_size must be positive and max_new nonnegative")
        unique = list(dict.fromkeys(texts))
        found: dict[str, NDArray[np.float64]] = {}
        missing = []
        for text in unique:
            cached = self.lookup(encoder, text)
            if cached is None:
                missing.append(text)
            else:
                found[text] = cached
        if not isinstance(encoder, HashingEncoder) and len(missing) > max_new:
            raise RuntimeError(f"Run needs {len(missing)} new API embeddings, exceeding --max-new={max_new}")
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            vectors = encoder.encode(chunk)
            if len(vectors) != len(chunk):
                raise RuntimeError("Encoder returned incorrect vector count")
            with self.connection:
                for text, vector in zip(chunk, vectors):
                    v = np.asarray(vector, dtype=np.float64)
                    if v.shape != (encoder.dimensions,) or not np.isfinite(v).all():
                        raise ValueError("Invalid encoder vector")
                    self.connection.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                        (self._key(encoder, text), encoder.name, encoder.dimensions,
                         v.astype("<f8").tobytes()),
                    )
                    found[text] = v
        return [found[text].copy() for text in texts]
