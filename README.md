# ABIDE Classifier

This project studies ASD vs control classification on ABIDE resting-state fMRI using two views of the same connectivity signal:

- Matrix features: vectorized Fisher z-transformed functional connectivity
- Graph features: sparse top-k graphs derived from the same FC matrices

The main notebook, [Pipeline.ipynb](Pipeline.ipynb), compares Logistic Regression, MLP, GCN, and GAT under one shared preprocessing and evaluation pipeline.

## What The Notebook Does

1. Builds subject-level FC matrices from CC200 ROI time series
2. Applies Fisher z-transform, zeroes the diagonal, and vectorizes the upper triangle for matrix models
3. Converts the same matrices into PyTorch Geometric graphs using top-k edge selection
4. Uses a site-holdout split, with UM_1 and YALE held out for final testing
5. Tunes models with 5-fold stratified cross-validation on the training split
6. Runs follow-up analyses on artifacts, preprocessing, graph construction, oversmoothing, and feature importance

## Data

- Dataset: ABIDE Preprocessed (FCP-INDI)
- Derivative: `rois_cc200` from the CPAC pipeline with `filt_global` preprocessing
- Final curated modeling set: 1035 subjects
- Site-holdout test set: 162 subjects from UM_1 and YALE

## Final Results

Final test-set performance reported in the notebook:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.7222 | 0.8006 |
| MLP | 0.6728 | 0.7679 |
| GCN | 0.6481 | 0.7026 |
| GAT | 0.6914 | 0.7220 |

The notebook’s final interpretation is that matrix-based models outperform graph models on this representation, while site confounds and global FC artifacts are not the dominant explanation. The most important signal appears to be sparse and edge-specific, which helps explain why message-passing GNNs underperform.

## Ablations And Checks

- Site-holdout evaluation versus the earlier random-split baseline
- Artifact analysis using FC distributions, summary statistics, and per-subject mean normalization
- Fisher z vs raw Pearson correlation comparison
- Graph construction ablations across sparsity and weighting choices
- GCN depth ablation to test oversmoothing
- Logistic Regression feature-importance analysis on the top FC edges

## Repository Layout

- `data_ingestion.ipynb`: ABIDE metadata filtering and time-series ingestion
- `pipeline.ipynb`: end-to-end modeling, tuning, evaluation, and ablations
- `data/abide_fmri/timeseries/`: ROI time-series files (`.1D`)
- `data/abide_fmri/connectivity_matrices/`: Fisher z FC matrices (`.npy`)
- `data/abide_fmri/pyg/`: PyTorch Geometric graph tensors (`.pt`)
- `data/abide_fmri/model_artifacts/`: saved models, hyperparameters, and analysis outputs
- `checkpoints/`: cached intermediate results for longer ablation runs

## Reproducibility Notes

- Random seed is fixed at `42`
- Model selection uses ROC-AUC on stratified 5-fold cross-validation
- The final evaluation is site-holdout, so the test score measures generalization to unseen acquisition sites rather than only unseen subjects from the same sites
- The notebook keeps legacy random-split comparisons only as a reference point

## Environment

The notebooks use Python with:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `torch-geometric`
- `matplotlib`
- `networkx`
- `tqdm`
- `requests`
