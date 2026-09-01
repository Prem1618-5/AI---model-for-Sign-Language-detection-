# Adversarial Stress Testing Report — Milestone M2
**Target Components**: Data Preprocessing & Leakage Isolation (`src/data_preprocessing.py`), Model Training & Serialization (`src/model_training.py`), Temporal Prediction Filter (`src/temporal_filter.py`).  
**Investigator**: `teamwork_preview_challenger` (Instance 2)  
**Date**: 2026-09-02  
**Overall Verdict**: **APPROVE** (All stress invariants satisfied, 0 regressions, 100% test pass rate across 81 unit & stress tests).

---

## Executive Summary

Milestone M2 focuses on ML training streamlining, data leakage prevention, model serialization fidelity, and temporal prediction stabilization. An empirical adversarial test suite (`tests/test_adversarial_m2_stress.py`) comprising 17 high-intensity stress probes was authored and executed alongside the standard 64-test suite (`tests/run_tests.py`).

All empirical tests passed cleanly with zero regressions. The data leakage fix was mathematically proven to prevent any augmented or unaugmented validation/test counterpart from leaking into the training set. Model training loops, batch size variations, early stopping mechanics, serialization round-trips, and temporal state transitions were exhaustively validated.

---

## 1. Challenge Dimension 1: Data Leakage & Partition Invariants

### 1.1 Code Path & Architecture
In `src/data_preprocessing.py`:
- `prepare_dataset()` extracts unaugmented raw landmark representations into `raw_samples`.
- Stratified partition splitting is performed on raw indices:
  1. `idx_trainval, idx_test = train_test_split(indices, y_all, test_size=0.20, ...)`
  2. `idx_train, idx_val = train_test_split(idx_trainval, y_trainval, test_size=val_size / (1.0 - test_size), ...)`
- `X_val` and `X_test` are sliced directly from unaugmented features (`X_all[idx_val]`, `X_all[idx_test]`).
- Data augmentation (`augment_landmarks()`) is executed strictly and exclusively for `idx in idx_train`.

### 1.2 Empirical Stress Harness & Mathematical Proof
- **Biometric Ratio Invariant**: Built synthetic landmark datasets where each raw gesture sample has a globally unique biometric ratio $R_k = \frac{\|\mathbf{p}_{\text{thumb\_tip}} - \mathbf{p}_{\text{wrist}}\|}{\|\mathbf{p}_{\text{pinky\_tip}} - \mathbf{p}_{\text{wrist}}\|}$.
- Because Euclidean joint distance ratios are strictly invariant under 2D/3D rigid rotations, translations, and uniform scaling ($0.9$ to $1.1$), any augmented sample derived from raw sample $k$ retains the exact ratio $R_k$.
- **Empirical Results**:
  - $100\%$ of augmented samples in $X_{\text{train}}$ matched a base sample in $X_{\text{train}}$ ($\Delta < 10^{-4}$).
  - $0\%$ of augmented samples in $X_{\text{train}}$ matched any sample in $X_{\text{val}}$ ($\min \Delta > 10^{-4}$) or $X_{\text{test}}$ ($\min \Delta > 10^{-4}$).
  - Strict disjointness verified: $X_{\text{train}} \cap X_{\text{val}} = \emptyset$, $X_{\text{train}} \cap X_{\text{test}} = \emptyset$, $X_{\text{val}} \cap X_{\text{test}} = \emptyset$.
  - Single-hand ($D=63$) and two-handed ($D=126$) feature dimensions verified without dimensional drift.

---

## 2. Challenge Dimension 2: Training Pipeline Robustness

### 2.1 Hyperparameter & Batch Size Stress Probes
In `src/model_training.py`:
- **Minimal Epochs (`epochs=1`)**: Model compiles, performs forward/backward pass, computes metrics, and persists artifacts without crashing or emitting NaN loss.
- **Batch Size Variations**:
  - `batch_size=1` (Stochastic Gradient Descent): Evaluated `BatchNormalization` and `Dropout` behaviors. Verified that post-training direct tensor evaluation and `predict_proba()` conserve probability distribution ($\sum P_i = 1.0$).
  - `batch_size=len(X_train)` (Full Batch GD): Successfully fits within memory and converges.
  - `batch_size > len(X_train)` (Oversized batch size: $B=500$ with $N=20$): Keras cleanly adapts step size without index boundary errors.
  - Odd/Prime Batch Sizes (`batch_size=3, 7, 13`): Handled uneven batch partitions without sample dropping or shape mismatch.
- **Early Stopping & Best Weights Restoration**:
  - Constructed synthetic dataset with deliberately divergent validation loss.
  - Verified `EarlyStopping(patience=10, restore_best_weights=True)` triggered before epoch limit ($100$ epochs) and restored the checkpoint with minimum validation loss.
  - Verified `best_model.h5` checkpoint persistence.
- **Extreme Class Imbalance**:
  - Tested class distribution with $10:1:1$ sample imbalance (50 samples vs 5 samples vs 5 samples). Training and classification report metrics completed without division by zero.
- **Architectures**:
  - Dense Single-Hand ($63 \to 128 \to 64 \to K$)
  - Dense Two-Handed ($126 \to 128 \to 64 \to K$)
  - Sequential LSTM ($63 \to \text{Reshape}(1, 63) \to \text{LSTM}(128) \to \text{LSTM}(64) \to \text{Dense}(64) \to K$)

---

## 3. Challenge Dimension 3: Serialization & Deserialization Round-Trip Fidelity

### 3.1 Model Persistence & Direct Tensor Inference Invariants
- **SavedModel Weight Invariance**:
  - Trained Dense and LSTM models, serialized via `save_model()`, and reloaded via `load_model()` into fresh instances.
  - All layer weight arrays compared across all parameters:
    $$\max_{l} \|W_{\text{original}}^{(l)} - W_{\text{reloaded}}^{(l)}\|_\infty = 0.0$$
- **Inference Numerical Invariance**:
  - Evaluated 100 arbitrary synthetic feature vectors through `model(tensor, training=False).numpy()`.
  - Max absolute difference between pre-save and post-reload predictions:
    $$\max |P_{\text{original}} - P_{\text{reloaded}}| < 10^{-6}$$
- **Metadata JSON Schema & Integrity**:
  - Verified `model_metadata.json` attributes: `model_type`, `class_names`, `input_shape`, `num_classes`, `is_two_handed`, `timestamp`.
  - Verified custom class name mapping persistence across reloads.
- **Directory Discovery & Error Handling**:
  - Verified auto-discovery of latest model in `model_dir`.
  - Verified explicit `model_path` loading.
  - Verified `FileNotFoundError` raised when model directory contains no valid model.

---

## 4. Challenge Dimension 4: Temporal Filter State Machine & Invariants

### 4.1 Filter Invariant Probes
In `src/temporal_filter.py`:
- **Softmax EMA Probability Conservation**: Verified $\sum_{i=1}^K S_t(i) = 1.0 \pm 10^{-5}$ over 500 consecutive random probability vectors.
- **Dual-Threshold Hysteresis Boundaries**:
  - $T_{\text{high}} = 0.80, T_{\text{low}} = 0.45, \text{debounce\_frames} = 4$.
  - Input $P = 0.79 < 0.80$ remains in `SCANNING` (`"None"`).
  - High confidence $P = 0.80$ for frames 1..3 stays in `UNCERTAIN` (`"Analysing..."`).
  - Frame 4 triggers transition to `DETECTED`.
  - Input dipping to $0.50$ or $0.45$ remains `DETECTED` (eliminating boundary flicker).
  - Input dropping to $0.40 < 0.45$ resets to `SCANNING`.
- **Kinematic Velocity Gating**:
  - Wrist displacement $> 0.08$ immediately resets debounce counter and forces status to `UNCERTAIN`.
  - Hand movement settling restores detection after debounce threshold.
  - Timestamp step $dt = 10^{-6}$ protected against ZeroDivisionError via `max(dt, 1e-4)`.
- **Buffer Bounds & Timeout**:
  - Sequence buffer strictly capped at 10 items.
  - Inactivity timeout ($> 1.5$s) clears sequence buffer and resets state.

---

## 5. Comprehensive Test Execution Summary

| Test Suite | File | Tests Run | Passed | Failed | Errors | Wall Time |
|---|---|---|---|---|---|---|
| M2 Adversarial Stress Suite | `tests/test_adversarial_m2_stress.py` | 17 | 17 | 0 | 0 | ~1.9s |
| Unified 5-Tier E2E Suite | `tests/run_tests.py` | 64 | 64 | 0 | 0 | ~15.1s |
| **Total Combined** | — | **81** | **81** | **0** | **0** | — |

---

## 6. Final Recommendation & Verdict

**Verdict: APPROVE**

The M2 implementation is robust, mathematically isolated from data leakage, fully tolerant to extreme batch size and hyperparameter variations, and maintains 100% serialization fidelity. Milestone M2 meets all architectural and quality criteria.
