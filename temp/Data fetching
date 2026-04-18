import os, requests, json, re
import pandas as pd

def fetch_data(target_n, save_dir, temp_min, temp_max):
    csv_path = os.path.join(save_dir, "metadata.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    all_ids = []
    for start in range(0, target_n * 5, 10000):
        query = {
            "query": {
                "type": "terminal", "service": "text",
                "parameters": {"attribute": "exptl_crystal_grow.temp", "operator": "exists"}
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": 10000}}
        }
        try:
            r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=60).json()
            ids = [h["identifier"] for h in r.get("result_set", [])]
            all_ids.extend(ids)
            if len(all_ids) >= target_n * 3: break
        except: continue

    records = []
    for i in range(0, len(all_ids), 100):
        if len(records) >= target_n * 1.5: break
        batch = all_ids[i:i + 100]
        gql = f"{{ entries(entry_ids: {json.dumps(batch)}) {{ rcsb_id exptl_crystal_grow {{ temp }} polymer_entities {{ entity_poly {{ pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }} }} }} }}"
        try:
            res = requests.post("https://data.rcsb.org/graphql", json={"query": gql}, timeout=60).json()
            for e in res.get("data", {}).get("entries", []):
                if not e.get("exptl_crystal_grow"): continue
                temp = e["exptl_crystal_grow"][0]["temp"]
                if temp is None: continue
                for poly in e.get("polymer_entities", []):
                    ep = poly["entity_poly"]
                    if "protein" not in str(ep.get("rcsb_entity_polymer_type", "")).lower(): continue
                    seq = re.sub(r"[^A-Z]", "", str(ep.get("pdbx_seq_one_letter_code_can", "")).upper())
                    if 40 < len(seq) < 500:
                        records.append({"pdb_id": e["rcsb_id"], "sequence": seq, "temp_k": float(temp)})
                        break
        except: continue

    df = pd.DataFrame(records).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    df = df[(df["temp_k"] >= temp_min) & (df["temp_k"] <= temp_max)].reset_index(drop=True)
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42).reset_index(drop=True)
    
    df.to_csv(csv_path, index=False)
    return df   

import os, re, torch, esm
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_esm(sequences, save_dir, batch_size):
    path = os.path.join(save_dir, "X_esm.npy")
    if os.path.exists(path): return np.load(path)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        _, _, tokens = batch_converter([("p", s) for s in batch])
        with torch.no_grad():
            results = model(tokens.to(device), repr_layers=[33])
        reps = results["representations"][33]
        for j, seq in enumerate(batch):
            embeddings.append(reps[j, 1:len(seq) + 1].mean(0).cpu().numpy())
        if i % 400 == 0: torch.cuda.empty_cache()

    X = np.array(embeddings, dtype=np.float32)
    np.save(path, X)
    return X

def compute_phys_single(seq, amino_acids):
    try:
        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "A", seq)
        pa = ProteinAnalysis(clean)
        counts = pa.get_amino_acids_percent()
        return np.array([pa.isoelectric_point(), pa.molecular_weight(), pa.gravy(), 
                         pa.aromaticity(), pa.instability_index()] + [counts.get(aa, 0.0) for aa in amino_acids], dtype=np.float32)
    except: return np.zeros(25, dtype=np.float32)

def build_feature_matrix(X_esm, sequences, amino_acids, pca_dims, save_dir):
    path = os.path.join(save_dir, "X_combined.npy")
    if os.path.exists(path): return np.load(path)

    X_phys = np.vstack([compute_phys_single(s, amino_acids) for s in sequences])
    X_phys = StandardScaler().fit_transform(X_phys).astype(np.float32)
    
    X_esm_scaled = StandardScaler().fit_transform(X_esm)
    X_pca = PCA(n_components=pca_dims, random_state=42).fit_transform(X_esm_scaled)
    
    X = np.hstack([X_pca, X_phys]).astype(np.float32)
    np.save(path, X)
    return X
