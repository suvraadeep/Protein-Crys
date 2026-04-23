"""
ESM-2 embedding extractor with disk-based caching.

First call per pdb_id runs the transformer and saves a 320-D float32
array to embeddings_cache/{pdb_id}.npy.  Subsequent calls (across all
four training scripts) load from disk — the transformer is never
re-run for a sequence already cached.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "embeddings_cache"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"   # 320-D, CPU-friendly


class ESMEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        device: str = "cpu",
        max_length: int = 1022,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading {self.model_name} onto {self.device} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
            print("ESM-2 loaded.")

    def _cache_path(self, pdb_id: str) -> Path:
        return self.cache_dir / f"{pdb_id}.npy"

    def _embed_single(self, sequence: str) -> np.ndarray:
        """Run ESM-2 on one sequence, return mean-pooled 320-D vector."""
        self._load_model()
        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length + 2,  # +2 for [CLS]/[EOS]
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # outputs.last_hidden_state: [1, seq_len+2, hidden]
        hidden = outputs.last_hidden_state[0]   # [seq_len+2, 320]
        # Mean-pool over residue positions, exclude [CLS] (0) and [EOS] (-1)
        embedding = hidden[1:-1].mean(dim=0).cpu().numpy()  # [320]
        return embedding.astype(np.float32)

    def embed_sequence(self, sequence: str, pdb_id: str) -> np.ndarray:
        """Return 320-D embedding; loads from cache if available."""
        path = self._cache_path(pdb_id)
        if path.exists():
            return np.load(str(path))
        emb = self._embed_single(sequence)
        np.save(str(path), emb)
        return emb

    def embed_batch(
        self,
        sequences: list[str],
        pdb_ids: list[str],
        batch_size: int = 4,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of sequences with caching.
        Returns ndarray of shape [N, 320].
        Sequences already cached are loaded from disk; only uncached
        ones are forwarded through ESM.
        """
        results: dict[int, np.ndarray] = {}
        uncached_idx, uncached_seqs, uncached_ids = [], [], []

        for i, (seq, pid) in enumerate(zip(sequences, pdb_ids)):
            path = self._cache_path(pid)
            if path.exists():
                results[i] = np.load(str(path))
            else:
                uncached_idx.append(i)
                uncached_seqs.append(seq)
                uncached_ids.append(pid)

        if uncached_seqs:
            self._load_model()
            n = len(uncached_seqs)
            if verbose:
                print(f"Embedding {n} uncached sequences (batch_size={batch_size}) ...")
            for b_start in range(0, n, batch_size):
                b_end = min(b_start + batch_size, n)
                b_seqs = uncached_seqs[b_start:b_end]
                b_ids = uncached_ids[b_start:b_end]
                if verbose and (b_start % (batch_size * 20) == 0):
                    print(f"  {b_start}/{n}")

                inputs = self._tokenizer(
                    b_seqs,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length + 2,
                )
                attention_mask = inputs["attention_mask"]
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)

                hidden = outputs.last_hidden_state.cpu()  # [B, L, 320]
                # Mean pool: exclude [CLS] and [EOS], respect padding
                for j in range(len(b_seqs)):
                    mask = attention_mask[j]  # [L]
                    h = hidden[j]             # [L, 320]
                    # positions 1 to (sum(mask)-2) are real residues
                    n_real = int(mask.sum().item()) - 2
                    if n_real <= 0:
                        n_real = 1
                    emb = h[1:1 + n_real].mean(dim=0).numpy().astype(np.float32)
                    global_idx = uncached_idx[b_start + j]
                    results[global_idx] = emb
                    np.save(str(self._cache_path(b_ids[j])), emb)

        embeddings = np.stack([results[i] for i in range(len(sequences))], axis=0)
        return embeddings   # [N, 320]

    def cache_all(
        self,
        sequences: list[str],
        pdb_ids: list[str],
        batch_size: int = 4,
    ):
        """Pre-build the full embedding cache (call once before training)."""
        print("=== Building embedding cache ===")
        self.embed_batch(sequences, pdb_ids, batch_size=batch_size, verbose=True)
        print("=== Cache complete ===")

    def cache_coverage(self, pdb_ids: list[str]) -> float:
        """Fraction of pdb_ids already cached."""
        cached = sum(1 for pid in pdb_ids if self._cache_path(pid).exists())
        return cached / len(pdb_ids) if pdb_ids else 1.0


if __name__ == "__main__":
    embedder = ESMEmbedder()
    test_seq = "MKVLSELDKAGITLGEMLPQVIAFYHKLYEEQAKREQAGDAASITASIASTLQSLIINTFYSNLEFESPENFQAAKIDELMQESQGGVVGIIKKCC"
    test_id = "TEST_SEQUENCE"
    emb = embedder.embed_sequence(test_seq, test_id)
    print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
    cache_file = embedder._cache_path(test_id)
    print(f"Cache file exists: {cache_file.exists()}")
