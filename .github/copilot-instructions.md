# Copilot Instructions for cross_hpc

This project performs **single-neuron decoding analysis** on calcium imaging data from mice running in a virtual reality environment with different contexts (rewarded vs non-rewarded trials).

## Data Overview
- **Source**: Two-photon calcium imaging experiments from hippocampus (HPC)
- **Experiments**: Loaded from `data.csv` which contains mouse IDs, trial IDs, and brain areas
- **Lab3 Framework**: Uses the `lab3` library for experiment management and signal processing

## Key Data Structures

### ICRWL Table
Behavioral summary per trial with columns:
- `context`: ctxA, ctxB (rewarded), ctxC, ctxD (non-rewarded)
- `reward_trial`: whether trial was a reward trial type
- `water`: whether water reward was delivered
- `lick_or_not`: whether mouse licked during trial

### Place Maps
Neural activity (df/f) organized as `(n_neurons, n_trials, n_position_bins)` - smoothed calcium signal at each spatial position

### AUC
Trial-summarized activity via trapezoidal integration: `(n_neurons, n_trials)`

## Analysis Pipeline

### 1. Trial-Summarized Single Neuron Decoding (`single_neuron_decoding`)
Uses logistic regression with 5-fold CV to decode 5 targets from each neuron's AUC:
- `context_ab`: Distinguish ctxA vs ctxB (both rewarded)
- `context_cd`: Distinguish ctxC vs ctxD (both non-rewarded)  
- `reward_trial`: Predict if trial was reward type
- `water`: Predict if water was delivered
- `lick_or_not`: Predict if mouse licked

Returns DataFrame with: `test_acc`, `test_auc`, `beta_mean`, `beta_std` per neuron per target

### 2. Temporal Single Neuron Decoding (`single_neuron_decoding_temporal`)
Time-resolved version using binned df/f (trapz over `bin_size` frames)
Returns xarray Dataset with dimensions `(neuron, target, time)` containing AUC, beta, intercept

## Visualization Functions
- `plot_auc()`: Heatmap of raw feature matrix
- `plot_class_balance()`: Horizontal bar showing class proportions for each target
- `plot_test_accuracy_by_neuron()`: Sorted scatter of mean accuracy per neuron
- `plot_pairwise()`: Target × Target pairwise accuracy scatter plots
- `plot_correlation_heatmap()`: Correlation between target decoding accuracies
- `plot_selectivity_histogram()`: Distribution of signed selectivity = (AUC - 0.5) × sign(β)
- `plot_neuron_beta_clusters()`: PCA + k-means clustering on beta coefficients
- `plot_temporal_encoding()`: Time-resolved encoding dynamics with SEM

## Key Files
- `utils.py`: Helper functions for data loading (`get_or_make_ICRWL`, `get_or_make_place_maps`), behavioral variable extraction, and decoding
- `data.csv`: Experiment metadata (mouse_id, trial_id, area)
- `01_single_cell.ipynb`: Main single-cell analysis notebook
- `00_s2p_viz.ipynb`: Suite2p visualization notebook

## Key Assumptions
- Contexts ctxA/ctxB are rewarded (`R_ctx`), ctxC/ctxD are non-rewarded (`NR_ctx`) - defined in `utils.py`
- Place maps may have more trials than behavioral data → need to crop to match
- Imaging data uses Suite2p df/f signals
