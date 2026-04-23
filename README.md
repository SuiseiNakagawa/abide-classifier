# ABIDE Classifier

This project studies ASD vs control classification on the [ABIDE Preprocessed dataset](http://preprocessed-connectomes-project.org/abide/index.html) resting-state fMRI using two views of the same connectivity signal:

- **Matrix features:** vectorized Fisher z-transformed functional connectivity (upper triangle)
- **Graph features:** sparse top-k graphs derived from the same FC matrices

Two parcellations are compared across the same pipeline: **CC200** (Craddock 200, data-driven, 200 ROIs) and **AAL** (Anatomical Automatic Labeling, anatomy-driven, 116 ROIs).

## Notebooks

| Notebook | Parcellation | ROIs | Features |
|---|---|---:|---:|
| [pipeline.ipynb](pipeline.ipynb) | CC200 (Craddock) | 200 | 19,900 |
| [aal.ipynb](aal.ipynb) | AAL | 116 | 6,670 |

Both use the same pipeline: site-holdout split, 5-fold CV tuning, and identical model architectures and ablations.

## What The Notebooks Do

1. Build subject-level FC matrices from ROI time series
2. Apply Fisher z-transform, zero the diagonal, and vectorize the upper triangle for matrix models
3. Convert the same matrices into PyTorch Geometric graphs using top-k edge selection
4. Use a site-holdout split, with UM_1 and YALE held out for final testing (162 subjects)
5. Tune models with 5-fold stratified cross-validation on the training split
6. Run follow-up analyses on artifacts, preprocessing, graph construction, oversmoothing, and feature importance

## Data

- Dataset: ABIDE Preprocessed (FCP-INDI)
- Derivatives: `rois_cc200` and `rois_aal` from the CPAC pipeline with `filt_global` preprocessing
- Final curated modeling set: 1035 subjects across 20 sites
- Site-holdout test set: 162 subjects from UM_1 (n=106) and YALE (n=56), perfectly balanced (50% ASD)
- Training set: 873 subjects from the remaining 18 sites

## Results

### CC200 (200 ROIs, 19,900 features)

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.7222 | **0.8006** |
| MLP | 0.7284 | 0.7754 |
| GCN | 0.6914 | 0.7715 |
| GAT | 0.6605 | 0.7235 |

### AAL (116 ROIs, 6,670 features)

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.6420 | 0.7452 |
| MLP | 0.6728 | **0.7487** |
| GCN | 0.6358 | 0.7234 |
| GAT | 0.6235 | 0.6839 |

## Parcellation Comparison

CC200 outperforms AAL across all four models (AUC gap: +2.7 to +5.5 pp):

| Model | CC200 AUC | AAL AUC | Δ |
|---|---:|---:|---:|
| Logistic Regression | 0.8006 | 0.7452 | +5.5 pp |
| MLP | 0.7754 | 0.7487 | +2.7 pp |
| GCN | 0.7715 | 0.7234 | +4.8 pp |
| GAT | 0.7235 | 0.6839 | +4.0 pp |

The model ranking also shifts: CC200 puts Logistic Regression clearly on top, while AAL closes the gap between LR and MLP (delta < 0.4 pp). The matrix-over-graph advantage holds in both parcellations.

**Feature signal character differs between parcellations:**

- CC200: sparse signal regime — >90% of LR coefficients have |coef| < 0.01; the top-20 edges account for a disproportionate share of total L2 norm
- AAL: diffuse signal regime — only 14.5% of LR coefficients fall below 0.01; the top-20 edges account for 3.6% of total L2 norm, with weight spread across many features

### Interpretation

CC200 is a *functionally*-defined parcellation: ROIs are derived by clustering voxels with similar BOLD time-series profiles. AAL is *anatomically*-defined: ROIs correspond to labeled gyri and sulci regardless of functional homogeneity. The consistent CC200 advantage plausibly reflects two interacting factors:

1. **Functional homogeneity within parcels.** CC200 ROIs capture regions that co-activate coherently, so the pairwise FC between two CC200 nodes more cleanly represents a functional pathway. AAL nodes may average across functionally heterogeneous subregions, diluting the connectivity signal that discriminates ASD from controls.

2. **Granularity confound.** CC200 has roughly 3× more features (19,900 vs 6,670). Some of the performance gap may come from the richer feature space rather than the parcellation principle per se. The two effects cannot be cleanly separated without a matched-resolution comparison.

The sparse vs diffuse signal finding adds a further observation: when CC200's data-driven boundaries produce parcels that are functionally coherent, a small number of specific connection pairs carry concentrated diagnostic weight. This is consistent with prior ASD neuroimaging literature suggesting atypical connectivity in specific circuits (e.g., default mode, salience, frontoparietal networks) rather than globally reduced or increased connectivity. AAL's anatomy-driven parcels appear to fragment this structured signal across more, weaker features — degrading both absolute performance and the sparse-signal structure that matrix models exploit.

**Caveats:** 

1. Both effects (parcellation principle and resolution) co-vary here; a 200-ROI anatomy-based atlas would be needed to isolate them. 
2. All results are from one preprocessing pipeline (CPAC, filt_global) and one train/test split; parcellation rankings may not generalise across pipelines or cohorts. 
3. The test set is 162 subjects — the 3–5 pp AUC differences are indicative but not individually significant at conventional thresholds.

## Ablations And Checks

Both notebooks include the same set of follow-up analyses:

- Artifact analysis: FC value distributions (KS tests), per-subject summary statistics, per-subject mean normalization ablation
- Fisher z vs raw Pearson correlation comparison
- Graph construction ablations: top-k=10, top-k=50, threshold=0.3, top-k=10 weighted
- GCN depth ablation (1-layer vs 2-layer) to test oversmoothing
- Logistic Regression feature-importance analysis on the top FC edges

## Repository Layout

```
pipeline.ipynb              CC200 pipeline (end-to-end modeling + ablations)
aal.ipynb                   AAL pipeline (same structure, PARCELLATION="aal")
data_ingestion.ipynb        ABIDE metadata filtering and time-series download

data/abide_fmri/
  subjects_clean.csv                    shared subject manifest (1035 subjects)
  subject_split_site_holdout.csv        shared train/test split
  train_subjects_site_holdout.csv
  test_subjects_site_holdout.csv
  cc200/
    timeseries/                         CC200 ROI time-series (.1D)
    connectivity_matrices/              Fisher z FC matrices (.npy)
    pyg/                                PyTorch Geometric graph tensors (.pt)
    model_artifacts/                    saved models and hyperparameters
    checkpoints/                        cached ablation results
    connectivity_index.csv
  aal/
    timeseries/                         AAL ROI time-series (.1D)
    connectivity_matrices/
    pyg/
    model_artifacts/
    checkpoints/
    connectivity_index.csv
```

## Reproducibility Notes

- Random seed fixed at `42`
- Model selection uses ROC-AUC on stratified 5-fold cross-validation
- Final evaluation is site-holdout: test score measures generalization to unseen acquisition sites
- Switch parcellations by changing `PARCELLATION = "cc200"` / `"aal"` in the cpu_setup cell

## Environment

```
numpy  pandas  scikit-learn  torch  torch-geometric  matplotlib  networkx  tqdm  requests
```
