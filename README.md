# ABIDE Graph Classifier

Classifying Autism Spectrum Disorder (ASD) vs control subjects from ABIDE preprocessed resting-state fMRI data using two representations of the same signal:

- Matrix features: vectorized Fisher z-transformed functional connectivity (FC)
- Graph features: top-k sparse graphs derived from FC matrices

The project compares four models under a shared data pipeline:

- Logistic Regression
- Multilayer Perceptron (MLP)
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)

## Project Goal

Probe whether graph-based inductive bias appears to help ASD classification over simpler matrix-based baselines when all models are built from the same ABIDE connectivity representation. The emphasis is on comparative behavior under one pipeline, not on building a diagnostic tool.

## Data Source

- Dataset: ABIDE Preprocessed (FCP-INDI)
- Derivative: `rois_cc200` (C-PAC, `filt_global` strategy)
- Subject filter:
	- Valid `FILE_ID` (`FILE_ID != "no_filename"`)
	- Diagnostic label in `{1, 2}` (ASD/control)
	- Available and valid CC200 time-series file

Final curated dataset used in the modeling notebook:

- Total subjects: 1035
- Train/test split (stratified by diagnosis): 828 / 207

## Pipeline Summary

1. Ingest phenotypic metadata and download required ROI time series (`data_ingestion.ipynb`)
2. Compute subject-level FC matrices from ROI time series using Pearson correlation
3. Apply Fisher z-transform, replace NaN/inf with 0, and zero diagonal
4. Build two representations:
	 - Matrix: upper triangle of FC (`19,900` features)
	 - Graph: top-k graph per subject (`k=10`, unweighted edges)
5. Train with 5-fold stratified cross-validation on training split
6. Evaluate tuned models on held-out test split

## Key Results (Test Set)

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.6667 | 0.7228 |
| MLP | 0.6715 | 0.7000 |
| GCN | 0.6329 | 0.6793 |
| GAT | 0.6087 | 0.6267 |

Majority-class baseline (train split): `51.2%` accuracy.

## Main Takeaways

- In this specific setup, matrix-based models performed better than graph-based models on ROC-AUC.
- Logistic Regression produced the strongest ranking performance, even as the simplest model.
- MLP achieved slightly higher accuracy, but not higher ROC-AUC than Logistic Regression.
- The current top-k graph construction did not show a clear advantage, which may reflect either weak graph signal or a graph formulation that does not align with the task.
- These results should be interpreted as pipeline-specific evidence rather than a general claim about ASD classification or graph neural networks.

## Repository Layout

- `data_ingestion.ipynb`: dataset curation and ABIDE derivative download
- `second_draft.ipynb`: FC construction, graph conversion, EDA, model training/evaluation
- `data/abide_fmri/timeseries/`: ROI time-series files (`.1D`)
- `data/abide_fmri/connectivity_matrices/`: Fisher z FC matrices (`.npy`)
- `data/abide_fmri/pyg/`: PyTorch Geometric graph tensors (`.pt`)
- `data/abide_fmri/model_artifacts/`: saved trained models and best hyperparameters

## Reproducibility Notes

- Random seed fixed to `42` in data splitting and CV routines.
- Cross-validation metric for tuning: ROC-AUC.
- Current split is diagnosis-stratified, not site-stratified.
	- Interpretation: results estimate generalization to new subjects under similar multi-site composition, not necessarily to entirely unseen sites.
- Functional connectivity is an indirect representation and depends on upstream preprocessing assumptions.

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
