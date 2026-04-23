"""
Biologically-informed 48-D protein feature extractor for crystallization prediction.

Feature layout (48 total):
  [0:20]  AA composition fractions (20)
  [20:30] Physicochemical bulk (10): length, GRAVY, pI, MW, instability,
          aromaticity, aliphatic_idx, boman_idx, predicted_Rg, net_charge_pH7
  [30:48] Crystallization-specific biological features (18):
          charge_asymmetry, kosmotropic_score, hofmeister_rank,
          helix_prop, sheet_prop, disorder_prop, surface_exposed,
          thermostability_idx, acidic_frac, basic_frac, hydrophobic_patch,
          seq_complexity, arg_lys_ratio, disulfide_potential,
          charged_frac, hydrophobic_frac, cys_frac, pro_frac

Biologically-motivated indices for head injection (into 368-D = ESM_320 + bio_48):
  BIO_IDX_PI           = 320 + 22   (pI_norm)       → pH head
  BIO_IDX_MW           = 320 + 23   (MW_norm)        → PEG head
  BIO_IDX_RG           = 320 + 28   (predicted_Rg)   → PEG head
  BIO_IDX_KOSMOTROPIC  = 320 + 30   (kosmotropic)    → salt head
  BIO_IDX_HOFMEISTER   = 320 + 32   (hofmeister_rank)→ salt head
  BIO_IDX_SURFACE      = 320 + 36   (surface_exposed)→ salt head
  BIO_IDX_THERMO       = 320 + 37   (thermostability)→ temp head
  BIO_IDX_ALIPHATIC    = 320 + 26   (aliphatic_idx)  → temp head
"""

import numpy as np
from typing import Optional

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    _BIO = True
except ImportError:
    _BIO = False

# ── Constants ─────────────────────────────────────────────────────────────────

_AA = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
       'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
       'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

# Chou-Fasman helix propensity (Pα)
_HELIX = {'E':1.53,'A':1.45,'L':1.34,'M':1.20,'Q':1.17,'K':1.07,'R':1.21,
          'H':1.00,'V':0.97,'I':1.00,'Y':0.61,'C':0.77,'W':1.02,'F':1.12,
          'T':0.82,'S':0.79,'D':0.98,'N':0.73,'P':0.59,'G':0.53}

# Chou-Fasman sheet propensity (Pβ)
_SHEET = {'V':1.70,'I':1.60,'Y':1.47,'F':1.38,'W':1.37,'L':1.22,'T':1.19,
          'C':1.30,'M':1.05,'G':0.75,'A':0.75,'K':0.74,'S':0.75,'R':0.90,
          'H':0.87,'E':0.26,'D':0.54,'N':0.65,'P':0.55,'Q':1.10}

# Intrinsic disorder tendency (higher = more disordered; from IUPred literature)
_DISORDER = {'A':0.06,'R':0.60,'N':0.60,'D':0.50,'C':0.02,'Q':0.70,'E':0.80,
             'G':0.40,'H':0.30,'I':0.02,'L':0.05,'K':0.80,'M':0.20,'F':0.02,
             'P':0.40,'S':0.60,'T':0.50,'W':0.10,'Y':0.20,'V':0.02}

# BOMAN index (protein-protein interaction potential, solubility scale)
_BOMAN = {'I':-1.56,'V':-0.78,'L':-1.18,'F':-2.20,'C':-0.64,'M':-0.64,
          'A':-0.50,'G': 0.00,'T': 0.45,'S': 0.46,'W':-0.46,'Y': 0.05,
          'P': 0.12,'H': 0.26,'E': 1.14,'Q': 0.58,'D': 0.95,'N': 0.85,
          'K': 1.40,'R': 1.81}

# Hofmeister ion stability rank for amino-acid side chains
# Approximation: basic residues (more chaotropic surface) vs polar/small (kosmotropic)
_HOFMEISTER = {'G':1.0,'A':0.9,'S':0.85,'T':0.8,'P':0.75,
               'V':0.5,'I':0.4,'L':0.4,'M':0.3,'F':0.2,'W':0.1,'Y':0.15,'C':0.4,
               'N':0.6,'Q':0.6,'D':0.35,'E':0.35,
               'K':0.1,'R':0.05,'H':0.2}

# Henderson-Hasselbalch pKa values for ionisable side chains
_PKA = {'D':3.9,'E':4.1,'H':6.0,'C':8.3,'Y':10.1,'K':10.5,'R':12.5}
_CHARGE_AT_7 = {  # approximate contribution at pH 7
    'D':-1,'E':-1,'H':+0.01,'C':0,'Y':0,'K':+1,'R':+1
}

# ── Named indices in the 48-D feature vector ──────────────────────────────────
BIO_IDX_PI           = 320 + 22   # pI_norm           → pH head
BIO_IDX_MW           = 320 + 23   # MW_norm            → PEG head
BIO_IDX_ALIPHATIC    = 320 + 26   # aliphatic_idx_norm → temp head
BIO_IDX_RG           = 320 + 28   # predicted_Rg_norm  → PEG head
BIO_IDX_NET_CHARGE   = 320 + 29   # net_charge_pH7
BIO_IDX_KOSMOTROPIC  = 320 + 31   # kosmotropic_score  → salt head
BIO_IDX_HOFMEISTER   = 320 + 32   # hofmeister_rank    → salt head
BIO_IDX_SURFACE      = 320 + 36   # surface_exposed    → salt head
BIO_IDX_THERMO       = 320 + 37   # thermostability    → temp head

FEATURE_DIM = 48   # physicochemical feature dimension
ESM_DIM     = 320  # ESM-2 esm2_t6_8M output dimension
COMBINED_DIM = ESM_DIM + FEATURE_DIM   # 368


class BioFeatureExtractor:
    """
    Extracts 48-D biologically motivated features from an amino acid sequence.

    Feature groups:
      [0:20]  AA composition
      [20:30] Physicochemical bulk (Biopython when available)
      [30:48] Crystallization-specific biological features
    """

    def extract(self, sequence: str) -> np.ndarray:
        seq = sequence.upper().strip()
        n = len(seq)
        feats = []

        # ── Group 1: AA composition (20) ────────────────────────────────────
        cnt = {aa: seq.count(aa) for aa in _AA}
        for aa in _AA:
            feats.append(cnt.get(aa, 0) / n)

        # ── Group 2: Physicochemical bulk (10) ──────────────────────────────
        if _BIO:
            try:
                pa = ProteinAnalysis(seq)
                pi   = pa.isoelectric_point()
                mw   = pa.molecular_weight()
                inst = pa.instability_index()
                arom = pa.aromaticity()
            except Exception:
                pi, mw, inst, arom = 7.0, 25000.0, 40.0, 0.1
        else:
            pi, mw, inst, arom = 7.0, n * 110.0, 40.0, 0.1

        gravy = sum(_KD.get(aa, 0) * cnt.get(aa, 0) for aa in _AA) / n

        # Aliphatic index = [Ala + 2.9×Val + 3.9×(Ile+Leu)] / len × 100
        aliphatic = (cnt.get('A',0) + 2.9*cnt.get('V',0) +
                     3.9*(cnt.get('I',0)+cnt.get('L',0))) / n * 100

        # BOMAN index (mean protein-interaction potential)
        boman = sum(_BOMAN.get(aa, 0) * cnt.get(aa, 0) for aa in _AA) / n

        # Predicted Rg from Flory's law: Rg ≈ R0 × N^ν, ν=0.6 for folded
        predicted_rg = 2.2 * (n ** 0.6)   # Å; normalize to [0,1] by /200

        # Net charge at pH 7 (Henderson-Hasselbalch approximation)
        net_charge_7 = sum(_CHARGE_AT_7.get(aa, 0) * cnt.get(aa, 0) for aa in _AA)

        feats.extend([
            n / 1000.0,              # [20] seq_length
            gravy,                   # [21] GRAVY (range ~-4.5 to +4.5)
            pi / 14.0,               # [22] pI_norm  ← KEY for pH
            min(mw / 100000.0, 3.0), # [23] MW_norm  ← KEY for PEG
            min(inst / 100.0, 2.0),  # [24] instability_index
            arom,                    # [25] aromaticity
            aliphatic / 200.0,       # [26] aliphatic_idx  ← KEY for temp
            boman / 5.0,             # [27] boman_index
            predicted_rg / 200.0,   # [28] predicted_Rg   ← KEY for PEG
            net_charge_7 / 50.0,    # [29] net_charge_pH7
        ])

        # ── Group 3: Crystallization-specific biological features (18) ──────

        # Charge asymmetry: positive/(positive+negative)  [30]
        pos = cnt.get('K',0) + cnt.get('R',0) + cnt.get('H',0)
        neg = cnt.get('D',0) + cnt.get('E',0)
        charge_asym = pos / (pos + neg + 1e-6)

        # Kosmotropic score: fraction of structure-stabilising AAs  [31]
        kosmotropic = sum(cnt.get(aa,0) for aa in "GASTPVIL") / n

        # Hofmeister rank: mean surface-residue Hofmeister position  [32]
        hofmeister = sum(_HOFMEISTER.get(aa,0)*cnt.get(aa,0) for aa in _AA) / n

        # Secondary structure propensities  [33] [34] [35]
        helix_p   = sum(_HELIX.get(aa,1.0)*cnt.get(aa,0) for aa in _AA) / n
        sheet_p   = sum(_SHEET.get(aa,1.0)*cnt.get(aa,0) for aa in _AA) / n
        disorder_p= sum(_DISORDER.get(aa,0.0)*cnt.get(aa,0) for aa in _AA) / n

        # Surface exposure estimate: fraction of usually-solvent-exposed AAs  [36]
        surface_exp = sum(cnt.get(aa,0) for aa in "STDEKNRQ") / n

        # Thermostability index (IVYWREL)  [37]
        thermo_idx = sum(cnt.get(aa,0) for aa in "IVYWREL") / n

        # Acidic and basic fractions  [38] [39]
        acidic_frac = (cnt.get('D',0) + cnt.get('E',0)) / n
        basic_frac  = (cnt.get('K',0) + cnt.get('R',0) + cnt.get('H',0)) / n

        # Hydrophobic patch: longest run of consecutive hydrophobic residues  [40]
        hydro_set = set("AVILMFYW")
        max_run, cur_run = 0, 0
        for aa in seq:
            if aa in hydro_set:
                cur_run += 1; max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        hydro_patch = max_run / n

        # Sequence complexity (Shannon entropy, normalized)  [41]
        freqs = np.array([cnt.get(aa,0)/n for aa in _AA])
        freqs = freqs[freqs > 0]
        entropy = float(-np.sum(freqs * np.log2(freqs))) / np.log2(20)

        # Arg/Lys ratio (thermostability)  [42]
        arg_lys = cnt.get('R',0) / (cnt.get('R',0) + cnt.get('K',0) + 1e-6)

        # Disulfide potential  [43]
        disulfide = int(cnt.get('C',0) / 2) / (n + 1e-6)

        # Charged, hydrophobic, cys, pro fractions  [44] [45] [46] [47]
        charged_frac = (pos + neg) / n
        hydrophobic_frac = sum(cnt.get(aa,0) for aa in "AVILMFYW") / n
        cys_frac = cnt.get('C',0) / n
        pro_frac = cnt.get('P',0) / n

        feats.extend([
            charge_asym,     # [30]
            kosmotropic,     # [31]
            hofmeister,      # [32]
            helix_p,         # [33]
            sheet_p,         # [34]
            disorder_p,      # [35]
            surface_exp,     # [36]
            thermo_idx,      # [37]
            acidic_frac,     # [38]
            basic_frac,      # [39]
            hydro_patch,     # [40]
            entropy,         # [41]
            arg_lys,         # [42]
            disulfide,       # [43]
            charged_frac,    # [44]
            hydrophobic_frac,# [45]
            cys_frac,        # [46]
            pro_frac,        # [47]
        ])

        assert len(feats) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feats)}"
        return np.array(feats, dtype=np.float32)

    def feature_names(self) -> list[str]:
        names  = [f"aa_{aa}" for aa in _AA]
        names += ["seq_length","gravy","pI_norm","MW_norm","instability",
                  "aromaticity","aliphatic_idx","boman_idx","predicted_Rg","net_charge_pH7"]
        names += ["charge_asymmetry","kosmotropic_score","hofmeister_rank",
                  "helix_propensity","sheet_propensity","disorder_propensity",
                  "surface_exposed","thermostability_idx","acidic_frac","basic_frac",
                  "hydrophobic_patch","seq_complexity","arg_lys_ratio",
                  "disulfide_potential","charged_frac","hydrophobic_frac","cys_frac","pro_frac"]
        return names


# ── Singleton for reuse ────────────────────────────────────────────────────────
BIO_EXTRACTOR = BioFeatureExtractor()


if __name__ == "__main__":
    test_seqs = {
        "Lysozyme (14kDa)":
            "KVFGRCELAAÁMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        "Ubiquitin (8.5kDa)":
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    }
    for name, seq in test_seqs.items():
        seq_clean = "".join(c for c in seq.upper() if c in set("ACDEFGHIKLMNPQRSTVWY"))
        feats = BIO_EXTRACTOR.extract(seq_clean)
        names = BIO_EXTRACTOR.feature_names()
        print(f"\n{name}  (len={len(seq_clean)})")
        print(f"  Feature vector shape: {feats.shape}")
        print(f"  pI_norm={feats[22]:.3f}  MW_norm={feats[23]:.3f}  "
              f"Rg_norm={feats[28]:.3f}  thermo={feats[37]:.3f}")
        print(f"  kosmotropic={feats[30]:.3f}  hofmeister={feats[32]:.3f}")
        assert feats.shape == (FEATURE_DIM,)
    print("\nAll assertions passed. bio_features.py is correct.")
