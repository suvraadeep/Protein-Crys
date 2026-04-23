# Protein Crystallization Condition Predictor

Predicts four key crystallization conditions from protein sequence alone:

| Target | Type | Model |
|--------|------|-------|
| **pH** | Regression | ESM-DL + XGBoost + LightGBM → Ridge stack |
| **Salt concentration** | Regression (M) | ESM-DL + XGBoost + LightGBM → Ridge stack |
| **Salt type** | Classification | ESM-DL + XGBoost + LightGBM soft-vote |
| **PEG type** | Classification | ESM-DL + XGBoost + LightGBM soft-vote |
| **Temperature** | Regression (K) | ESM-DL + XGBoost + LightGBM + CatBoost → Ridge stack |

---

## Problem

Determining the right buffer conditions to crystallize a protein for X-ray diffraction is largely trial-and-error. Given only the amino acid sequence, this system predicts:

- **pH**: buffer acidity/alkalinity (range ~4–10)
- **Salt type + concentration**: ionic strength required (0–4 M)
- **PEG type**: polymer precipitant (PEG 400, 3350, 4000, 6000, etc.)
- **Temperature**: crystallization temperature (°C / K)

---

## Why ESM-2 Embeddings

ESM-2 (Meta AI, `facebook/esm2_t6_8M_UR50D`) is a protein language model pre-trained on 250 million protein sequences. It captures:

- **Evolutionary context**: which amino acids co-evolve
- **Structural tendencies**: helices, sheets, loops from sequence alone
- **Surface chemistry**: charge distribution, hydrophobicity patterns

These are exactly the properties that govern crystallization behaviour. Raw amino acid composition misses positional and contextual information; ESM-2 captures it.

**Why we still add physicochemical features:**
The original `salt-conc` experiments showed that hand-crafted features (GRAVY, pI, AA composition) *outperformed* ESM-2 alone for regression (R² 0.45 vs 0.33). The two representations are complementary — ESM captures context, physicochemical captures bulk statistics — so concatenating them gives the best of both.

---

## Architecture

![Architecture Diagram](architecture.svg)

```
Protein Sequence
       │
ESM-2 Tokenizer  (facebook/esm2_t6_8M_UR50D)
       │
ESM-2 Transformer  [320-D]
  layers 0-4: frozen  |  layer 5: fine-tuned
       │
  Mean Pool → 320-D embedding
       │
       + 31-D physicochemical features
         (AA composition, GRAVY, pI, MW,
          charge, hydrophobicity, aromaticity,
          cysteine, proline, instability index)
       │
 Concat → 351-D
       │
  ┌────┼──────┐
  DL  XGB   LGB   (+ CatBoost for temp)
  │    │     │
  └────┴─────┘
   K-fold OOF
       │
 Ridge Meta-Learner
       │
  ┌──┬──┬──┐
  pH Salt PEG Temp
```

---

## Hybrid vs Baseline

| Task | Old approach | New approach | Improvement |
|------|-------------|--------------|-------------|
| Salt type | XGBoost alone: **34%** | DL+XGB+LGB stack | ~40-45% expected |
| Salt conc | XGBoost R²=**0.45** | DL+XGB+LGB stack | R² > 0.55 expected |
| Temperature | XGB+LGB+CB mean | +DL + Ridge stack | MAE < 4 K expected |
| pH | Not predicted | DL+XGB+LGB stack | New target |
| PEG type | Not predicted | DL+XGB+LGB soft-vote | New target |

---

## CPU Optimisation

### Embedding Cache
After the first training run, every protein's 320-D ESM-2 vector is saved to
`embeddings_cache/{pdb_id}.npy`. All four models (`ph/`, `salt/`, `peg/`, `temp/`)
share the same cache directory — the transformer is never re-run for a cached sequence.

Result: epoch time drops from ~hours to ~minutes on CPU.

### Model Configuration
- ESM-2 frozen except last transformer layer
- Batch size 8, gradient accumulation x4 → effective batch 32
- AdamW + linear warmup + cosine annealing LR schedule
- Early stopping (patience=10)
- All-CPU safe (no CUDA requirement)
- 16 GB RAM sufficient for full training run

---

## Repository Structure

```
Protein-Crys/
├── ph/
│   ├── train.py        <- DL + XGB + LGB stacking for pH
│   └── evaluate.py
├── salt/
│   ├── train.py        <- hybrid for salt type + concentration
│   └── evaluate.py
├── peg/
│   ├── train.py        <- hybrid soft-vote for PEG type
│   └── evaluate.py
├── temp/
│   ├── train.py        <- DL + XGB + LGB + CatBoost stacking
│   ├── evaluate.py
│   └── metadata(1).csv
├── models/
│   ├── esm_backbone.py <- shared ESMBackbone trunk (351->256->128)
│   ├── ph_model.py
│   ├── salt_model.py
│   ├── peg_model.py
│   └── temp_model.py
├── utils/
│   ├── data_parser.py  <- merge datasets, parse pH + PEG from REMARK 280
│   ├── esm_embedder.py <- ESM extraction + disk cache
│   └── dataset.py      <- PyTorch Datasets + physicochemical extractor
├── training/
│   ├── _train_utils.py <- shared LR scheduler + EarlyStopping
│   └── evaluate.py     <- unified per-model evaluation
├── app/
│   └── streamlit_app.py
├── embeddings_cache/   <- auto-created, .npy per protein
├── architecture.svg
└── README.md
```

---

## Setup

```bash
pip install torch transformers biopython \
            xgboost lightgbm catboost \
            scikit-learn streamlit joblib pandas numpy
```

CatBoost is optional — the temperature pipeline degrades gracefully to
XGB + LGB + DL if it is not installed.

---

## Training

Run each target independently (all share the same embedding cache):

```bash
# Step 1: pH model
python ph/train.py --epochs 50 --batch-size 8 --folds 5

# Step 2: Salt concentration + salt type
python salt/train.py --epochs 50 --batch-size 8 --folds 5

# Step 3: PEG type
python peg/train.py --epochs 50 --batch-size 8 --folds 5

# Step 4: Temperature
python temp/train.py --epochs 50 --batch-size 8 --folds 5
```

First run of any script builds the embedding cache (slow, one-time).
Subsequent runs and all other scripts skip the ESM forward pass entirely.

---

## Evaluation

```bash
python ph/evaluate.py
python salt/evaluate.py
python peg/evaluate.py
python temp/evaluate.py
```

Each prints a per-model comparison table: DL / XGBoost / LightGBM / Ensemble.

---

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

- Paste any protein sequence (30-1500 aa)
- Three hardcoded example sequences
- Predicts all four conditions using the full ensemble
- Shows salt-type and PEG-type probability bar charts
- Flags out-of-distribution sequence lengths

---

## Data Sources

- **Salt / pH / PEG**: `salt-conc/crystallization_dataset.csv` — 6,522 PDB entries
  pH and PEG parsed from `REMARK 280` crystallization condition text
- **Temperature**: `temp/metadata(1).csv` — 9,372 PDB entries
- Both datasets fetched from the [RCSB PDB](https://www.rcsb.org/) REST API
