import os, re
import numpy as np
import torch
import esm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PCA_DIMS = 200
BATCH_SIZE = 8

def extract_esm(sequences, save_dir):
    path = f"{save_dir}/X_esm.npy"
    if os.path.exists(path):
        return np.load(path)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()
    embeddings = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i:i+BATCH_SIZE]
        _, _, tokens = batch_converter([("p", s) for s in batch])

        with torch.no_grad():
            results = model(tokens.to(device), repr_layers=[33])

        reps = results["representations"][33]

        for j, seq in enumerate(batch):
            embeddings.append(
                reps[j, 1:len(seq)+1].mean(0).cpu().numpy()
            )

    X = np.array(embeddings, dtype=np.float32)
    np.save(path, X)
    return X


def compute_phys(seq):
    try:
        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "A", seq)
        pa = ProteinAnalysis(clean)

        feats = [
            pa.isoelectric_point(),
            pa.molecular_weight(),
            pa.gravy(),
            pa.aromaticity(),
            pa.instability_index()
        ]

        counts = pa.get_amino_acids_percent()
        feats += [counts.get(aa, 0.0) for aa in AMINO_ACIDS]

        return np.array(feats, dtype=np.float32)
    except:
        return np.zeros(25, dtype=np.float32)


def build_phys_matrix(sequences, save_dir):
    path = f"{save_dir}/X_phys.npy"
    if os.path.exists(path):
        return np.load(path)

    X = np.vstack([compute_phys(s) for s in sequences])
    X = StandardScaler().fit_transform(X).astype(np.float32)

    np.save(path, X)
    return X


def build_feature_matrix(X_esm, X_phys, save_dir):
    path = f"{save_dir}/X_combined.npy"
    if os.path.exists(path):
        return np.load(path)

    X_scaled = StandardScaler().fit_transform(X_esm)
    X_pca = PCA(n_components=PCA_DIMS, random_state=42).fit_transform(X_scaled)

    X = np.hstack([X_pca, X_phys]).astype(np.float32)
    np.save(path, X)

    return X
