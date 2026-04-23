# CrystalNet Architecture & Biology Reference

## 1. System Overview

Predicting protein crystallization conditions from sequence alone is a hard inverse-design problem.
The crystallization space is high-dimensional (pH, salt type, salt concentration, PEG type/concentration,
temperature) and the relationship between protein sequence and optimal conditions is non-linear,
dependent on subtle surface chemistry, and historically captured only through empirical screens.

CrystalNet addresses this with a **hybrid stacking ensemble** that combines:
- A **deep learning trunk** (ESM-2 transformer + MLP) that learns from evolutionary co-variation
  encoded across 250 million protein sequences
- **Gradient-boosted trees** (XGBoost + LightGBM + CatBoost) that exploit tabular structure
  efficiently on CPU
- A **biologically grounded 48-D feature vector** that encodes known crystallization biochemistry
  directly (Hofmeister, pI, Flory radius, thermostability indices)
- A **Ridge meta-learner** trained on out-of-fold predictions that learns optimal blending weights

Per-target predictions are produced by four independent pipelines sharing one backbone and one
embedding cache, each with a **biologically-guided head injection** (see §6).

---

## 2. ESM-2 Biology — What the Transformer Learned

ESM-2 (`facebook/esm2_t6_8M_UR50D`) is a masked-language model pre-trained on UniRef50 (250M
sequences) using a BERT-style objective: predict masked amino acids from context.

Key properties relevant to crystallization prediction:

- **Evolutionary coupling captured in attention heads.** Residue pairs that co-evolve (because
  mutations in one must be compensated by mutations in the other to maintain function) produce
  high mutual information in the MSA — the same signal is distilled into attention weights during
  pre-training on the full database without explicit MSA alignment.
- **Surface exposure implicit in layer activations.** Buried hydrophobic residues receive very
  different contextual embeddings from solvent-exposed ones; this directly encodes the surface
  chemistry that drives crystal contacts.
- **Mean-pool over positions [1:-1]** (excluding [CLS]/[EOS] special tokens) produces a 320-D
  protein-level vector that summarises evolutionary constraints across all positions.

In CrystalNet, layers 0–4 are **frozen** (pretrained general features) and layer 5 is
**fine-tuned** via backpropagation through the DL base learner only. This avoids catastrophic
forgetting while allowing the model to shift attention weights toward crystallization-relevant
residues.

---

## 3. 48-D Biological Feature Vector

The 48-D vector is divided into three groups:

### Group 1 — Amino Acid Composition [0:20]
Fraction of each of the 20 standard amino acids.  Simple but captures overall residue type budget.

### Group 2 — Physicochemical Bulk Properties [20:30]

| Index | Feature | Scale | Biology |
|-------|---------|-------|---------|
| 20 | `seq_length` | /1000 | Larger proteins need higher-MW PEG |
| 21 | `GRAVY` | raw (≈-4 to +4) | Kyte-Doolittle: positive = hydrophobic core |
| 22 | `pI_norm` | /14 | **Key for pH** — see §4 |
| 23 | `MW_norm` | /100 kDa, capped 3 | **Key for PEG** — see §5 |
| 24 | `instability_index` | /100 | Ikai 1980; >40 = unstable |
| 25 | `aromaticity` | fraction | F+W+Y fraction; packing contacts |
| 26 | `aliphatic_idx` | /200 | **Key for temp** — see §7 |
| 27 | `boman_index` | /5 | Protein-protein interaction potential |
| 28 | `predicted_Rg` | /200 Å | Flory N^0.6 estimate; **Key for PEG** |
| 29 | `net_charge_pH7` | /50 | Henderson-Hasselbalch at pH 7 |

### Group 3 — Crystallization-Specific Biological Features [30:47]

| Index | Feature | Biology rationale |
|-------|---------|-------------------|
| 30 | `charge_asymmetry` | pos/(pos+neg); asymmetry → selective crystal contacts |
| 31 | `kosmotropic_score` | Fraction G,A,S,T,P,V,I,L → salting-out tendency |
| 32 | `hofmeister_rank` | Weighted Hofmeister position; predicts which salt |
| 33 | `helix_propensity` | Chou-Fasman Pα mean; α-helical proteins crystallize differently |
| 34 | `sheet_propensity` | Chou-Fasman Pβ; β-rich → often need PEG 4000+ |
| 35 | `disorder_propensity` | IUPred-like score; high disorder → harder to crystallize |
| 36 | `surface_exposed` | Fraction S,T,D,E,K,R,N,Q; hydration layer thickness |
| 37 | `thermostability_idx` | IVYWREL fraction; **Key for temp** |
| 38 | `acidic_frac` | D+E; lowers pI; abundant in acidic-pH crystallizers |
| 39 | `basic_frac` | K+R+H; raises pI; abundant in alkaline-pH crystallizers |
| 40 | `hydrophobic_patch` | Longest hydrophobic run / length; aggregation risk |
| 41 | `seq_complexity` | Shannon entropy/log2(20); low = repetitive → easier crystal |
| 42 | `arg_lys_ratio` | Arg/(Arg+Lys); Arg donates more H-bonds → thermostability |
| 43 | `disulfide_potential` | int(Cys/2)/len; disulfides constrain packing options |
| 44 | `charged_frac` | (pos+neg)/len; high charge → needs higher ionic strength |
| 45 | `hydrophobic_frac` | AVILMFYW fraction; drives core packing |
| 46 | `cys_frac` | Cys/len; redox sensitivity affects condition choice |
| 47 | `pro_frac` | Pro/len; Pro disrupts secondary structure; affects packing |

---

## 4. The pI–pH Relationship

The isoelectric point (pI) is the pH at which a protein's net charge is zero.  Near the pI:

1. Electrostatic repulsion between protein molecules is minimised.
2. The energy barrier to nucleation of the crystal lattice is lowest.
3. The protein is least soluble, so lower precipitant concentration is needed.

**Henderson-Hasselbalch model** used in CrystalNet:

```
net_charge(pH) = Σ [charge_i × f_i(pH)]
```

where `f_i` is the fractional ionisation of residue type `i` computed from the pKa value.
The pI is found where `net_charge = 0`.

The **PHModel** head receives `trunk_128 | pI_norm` as a 129-D vector before the final
linear layers.  This explicit injection of pI provides a direct anchor even if the ESM
embedding has not fully captured the pI signal.

---

## 5. PEG Excluded-Volume Theory

PEG drives crystallization by the **depletion interaction**:
PEG polymers are sterically excluded from the volume immediately surrounding the protein.
This creates an unequal osmotic pressure — lower in the protein-depleted zone — that pushes
protein molecules together, effectively increasing the chemical potential of the
protein-protein contact state.

The magnitude of this effect depends on **polymer-to-protein size ratio**:
- If PEG MW >> protein MW: too much excluded volume, protein may precipitate amorphously
- If PEG MW << protein MW: insufficient depletion force, no crystallization
- Optimal: PEG Rg ≈ protein Rg → matched excluded volume

**Flory scaling law** (used in feature index 28):
```
Rg ≈ R₀ × N^ν
```
where `N` is sequence length, `ν ≈ 0.6` for folded globular proteins,
and `R₀ ≈ 2.2 Å` from calibration on known crystal structures.

The **PEGModel** head injects `predicted_Rg` and `MW_norm` alongside the 128-D trunk.

---

## 6. Per-Target Biologically-Guided Head Injection

Each model head receives the 128-D trunk output **concatenated with a small set of
biologically relevant features** extracted directly from the 368-D combined input:

| Target | Injected features | Rationale |
|--------|-------------------|-----------|
| pH | `pI_norm` (1-D) | pH ≈ pI ± δ |
| Salt conc. | `kosmotropic_score`, `hofmeister_rank`, `surface_exposed` (3-D) | Direct Hofmeister series encoding |
| PEG type | `predicted_Rg`, `MW_norm` (2-D) | Depletion interaction MW matching |
| Temperature | `thermostability_idx`, `aliphatic_idx` (2-D) | IVYWREL + aliphatic dual-index |

This means the head linear layers are 129-, 131-, 130-, and 130-D wide respectively.
The trunk (128-D) provides the global sequence context; the injected features anchor
the prediction to known biochemical constraints.

---

## 7. Thermostability Indices

Two sequence-derived indices predict crystallization temperature:

### IVYWREL index
Derived from comparative proteomics of thermophilic vs. mesophilic organisms:
```
IVYWREL = (I + V + Y + W + R + E + L) / sequence_length
```
Thermophilic proteins (optimally active at 50–80 °C) are enriched in these residues
because they provide hydrophobic core stability, aromatic stacking, and bifurcated
H-bond donation (Arg).

### Aliphatic index
```
AI = (Ala + 2.9×Val + 3.9×(Ile+Leu)) / len × 100
```
Introduced by Ikai (1980) as a volume-based thermostability proxy.  Branched aliphatic
side chains (Val, Ile, Leu) contribute more to hydrophobic core packing per residue than
Ala.  AI > 80 correlates with stability at room temperature; AI > 100 with thermostability.

---

## 8. Stacking Ensemble Rationale

A simple average of base-learner predictions is suboptimal because:
- DL models tend to smooth predictions (low variance, can be biased near the boundary)
- XGBoost/LightGBM models fit fine-grained feature interactions but can overfit
- Their error profiles are partially complementary

**K-fold OOF (Out-Of-Fold) stacking:**
1. Split training data into K folds
2. For each fold: train base learners on K-1 folds, predict on the held-out fold
3. Collect all OOF predictions → `[N, n_base_learners]` meta-feature matrix
4. Train Ridge (L2-regularised linear regression) or LogisticRegression on this matrix
5. The Ridge coefficients learn per-learner blending weights without overfitting

This is strictly superior to simple averaging and to a holdout-only meta-learner
because all N samples contribute OOF predictions without data leakage.

---

## 9. CPU Optimisation

**Embedding cache math:**
- ESM-2 forward pass on CPU for a 300-aa protein: ~0.8 s
- With 6500 proteins × 0.8 s = 1.45 hours if not cached
- With cache: each embedding is read from disk in ~0.5 ms
- Cache speedup factor: ~1600×

The cache stores `{pdb_id}.npy` (320-D float32 = 1.28 KB per protein).
6500 proteins × 1.28 KB = ~8.3 MB total cache size.

**Training memory budget:**
- Batch size 8 × effective 32 (via gradient accumulation, accum_steps=4)
- Model parameter count: ~500 K (ESMBackbone + head)
- Peak RAM during training: ~2 GB (dominated by ESM-2 model weights, ~30 MB)

---

## 10. Architecture Diagrams

See the `svg/` folder for per-target and full-system diagrams:

| File | Description |
|------|-------------|
| [svg/full_arch.svg](svg/full_arch.svg) | Complete CrystalNet stacking system |
| [svg/ph_arch.svg](svg/ph_arch.svg) | pH prediction: pI-injected head |
| [svg/salt_arch.svg](svg/salt_arch.svg) | Salt: Hofmeister-injected head |
| [svg/peg_arch.svg](svg/peg_arch.svg) | PEG: Flory Rg–MW injected head |
| [svg/temp_arch.svg](svg/temp_arch.svg) | Temperature: IVYWREL + aliphatic injected head |

---

## References

- Rives et al. (2021) "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *PNAS* 118(15)
- Hofmeister F. (1888) "Zur Lehre von der Wirkung der Salze." *Arch. Exp. Pathol. Pharmakol.*
- Arakawa & Timasheff (1985) "Theory of protein solubility." *Methods Enzymol.* 114:49–77
- Ikai A. (1980) "Thermostability and aliphatic index of globular proteins." *J. Biochem.* 88:1895–1898
- Flory P.J. (1953) *Principles of Polymer Chemistry.* Cornell University Press
- Matthews B.W. (1968) "Solvent content of protein crystals." *J. Mol. Biol.* 33:491–497
- Chou & Fasman (1978) "Prediction of secondary structure of proteins from amino acid sequence." *Adv. Enzymol.* 47:45–148
