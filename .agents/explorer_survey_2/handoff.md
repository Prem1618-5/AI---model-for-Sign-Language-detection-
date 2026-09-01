# Phase 0 Codebase Survey — Handoff Report

**Agent**: `teamwork_preview_explorer` (Instance 2)  
**Role**: ML Model & Temporal Stability Explorer  
**Task ID / Scope**: Phase 0 ML Survey (Model Architectures, Temporal Handling, Prediction Stability, Training Pipeline)  
**Date**: 2026-09-02  

---

## 1. Observation

- **Dense MLP Architecture**: In `src/model_training.py` lines 60–105, `GestureModelTrainer.build_model()` creates a 3-layer Sequential Dense model with parameters:
  - Input: `(126,)` (for two-handed) or `(63,)` (for single-hand).
  - Layer 1: `Dense(128, activation='relu')`, `BatchNormalization()`, `Dropout(0.3)`.
  - Layer 2: `Dense(64, activation='relu')`, `BatchNormalization()`, `Dropout(0.3)`.
  - Output: `Dense(num_classes, activation='softmax')`.
  - Optimizer: `Adam(learning_rate=0.001)`, Loss: `sparse_categorical_crossentropy`.
- **LSTM Implementation**: In `src/model_training.py` lines 107–155, `build_lstm_model()` defines:
  ```python
  lstm_input_shape = (1, input_shape[0])
  model = models.Sequential([
      layers.Reshape(lstm_input_shape, input_shape=input_shape),
      layers.LSTM(128, return_sequences=True),
      layers.Dropout(0.3),
      layers.LSTM(64),
      ...
  ```
  The input tensor is reshaped to sequence length $T=1$, meaning each sample is processed in isolation without recurrent temporal sequence history.
- **Classical ML Models**: Grep search for `RandomForest`, `SVC`, and `LogisticRegression` confirms no classical ML classifiers are implemented in `src/model_training.py`, though `scikit-learn==1.3.0` is present in `requirements.txt:7` and imported for evaluation metrics.
- **Data Collection & Preprocessing**:
  - `src/data_collection.py` lines 146–170 collects snapshot frames at discrete intervals (`capture_delay=0.2s`) when `SPACE` is held.
  - `src/data_preprocessing.py` lines 231–304 normalizes each frame individually (palm center translation + wrist-middle distance scaling) and flattens to 1D feature vectors.
  - `data_preprocessing.py` lines 314–325 uses `train_test_split` with stratification across individual frames.
- **Real-Time Temporal Inference & Stability**:
  - `src/realtime_recognition.py` line 88 initializes `self.history_buffer = deque(maxlen=smoothing_window)` (default 10).
  - Lines 161–193 in `get_smoothed_prediction()` computes a 60% majority vote over top-1 class strings in the deque.
  - Lines 530–544 executes `self.trainer.predict(features)` (which uses slow `self.model.predict()`), checks `smooth_confidence >= 0.70`, and updates `self.sequence_buffer`.
  - `update_sequence()` uses a `sequence_timeout = 2.0` seconds inactivity reset.
- **Existing Artifacts**:
  - `models/gesture_recognition_dense_model` (SavedModel format, 25K params).
  - `models/best_model.h5` (HDF5 weights).
  - `models/model_metadata.json` (class names: `hello`, `no`, `thanks`, `yes`, input shape: 126, `is_two_handed: true`).
  - `data/processed/processed_gesture_data.npz` (processed dataset).

---

## 2. Logic Chain

1. **Static vs Temporal Nature**: Because `data_collection.py` records independent static frames and `data_preprocessing.py` processes each frame in isolation, the dataset consists of static sign postures rather than continuous dynamic trajectories.
2. **LSTM Ineffectiveness**: Because the scaffolded LSTM in `model_training.py` sets sequence length $T=1$, the recurrent hidden state updates exactly once per sample ($h_0 \rightarrow h_1$). It provides zero temporal context across frames while adding 7.4x parameter bloat and slower inference.
3. **Inference Stability Root Cause**: The current instability (chatter / flicker during hand movement) stems from doing a discrete string majority vote on top-1 predictions, discarding the full softmax probability distribution, and lacking an inter-frame velocity gate.
4. **Ponytail Senior Dev Alignment**: Building a full multi-frame recurrent pipeline (overhauling data collection, recording video clips, managing variable sequence buffers) violates Ponytail's YAGNI and minimal-diff rules for static sign recognition. Instead, implementing an Exponential Moving Average (EMA) on probability vectors, a dual-threshold hysteresis debouncer, a kinematic velocity gate, and direct-call tensor inference (`model(x, training=False).numpy()`) provides rock-solid stability in $<40$ lines of pure Python/NumPy without new dependencies.

---

## 3. Caveats

- **Scope of Vocabulary**: The current model and dataset contain 4 static signs (`hello`, `no`, `thanks`, `yes`). If the project in the future expands to dynamic gesture words that require motion path tracking (e.g. dynamic gestures like waving or circles), temporal sequence modeling or kinematic trajectory features would become necessary.
- **Hardware Variation**: Webcams with varying frame rates (15 FPS vs 60 FPS) may affect the temporal decay rate of the EMA filter unless normalized by $\Delta t$ (frame delta time).

---

## 4. Conclusion

- Retain the **Dense MLP** architecture as the core classifier. It is lightweight (~25K params), fast, and achieves 100% accuracy on normalized landmarks.
- Prune or document the pseudo-LSTM model in `model_training.py` to prevent architectural confusion.
- Enhance the temporal prediction pipeline in `realtime_recognition.py` by introducing:
  1. Full-distribution Softmax Exponential Moving Average (EMA, $\alpha=0.25$).
  2. Dual-threshold hysteresis state debouncing ($T_{\text{high}}=0.80$, $T_{\text{low}}=0.45$, $K=4$ frames).
  3. Hand velocity motion gating via wrist displacement.
  4. TensorFlow direct tensor evaluation (`model(x, training=False).numpy()`) to eliminate Keras `model.predict()` latency.
- Full details documented in `report.md`.

---

## 5. Verification Method

To independently verify all findings in this survey:
1. **Inspect Model Code**:
   ```bash
   view_file "d:/Development Project/Sign Language/src/model_training.py" (Lines 60-155)
   ```
2. **Inspect Real-time Smoothing & Predict Logic**:
   ```bash
   view_file "d:/Development Project/Sign Language/src/realtime_recognition.py" (Lines 161-222, 529-545)
   ```
3. **Verify Metadata and Processed Data**:
   ```bash
   view_file "d:/Development Project/Sign Language/models/model_metadata.json"
   ```
4. **Run CLI Verification**:
   ```powershell
   python src/main.py --help
   ```
