# Phase 0 Codebase Survey: Sign Language Detection ML System

**Date**: 2026-09-02  
**Author**: `teamwork_preview_explorer` (Instance 1)  
**Target Repository**: `d:\Development Project\Sign Language`  
**Working Directory**: `d:\Development Project\Sign Language\.agents\explorer_survey_1`

---

## 1. Executive Summary

The Sign Language Detection ML system is an end-to-end computer vision and machine learning pipeline built on **OpenCV**, **Google MediaPipe**, and **TensorFlow / Keras**. Its objective is to capture real-time webcam video, extract 3D hand landmarks (21 keypoints per hand), classify gestures across single-hand and dual-hand vocabularies, and render a styled native OpenCV desktop Heads-Up Display (HUD).

### Key Survey Findings:
1. **Pipeline Completeness**: The core functional loop—Data Collection (`data_collection.py`) $\rightarrow$ Preprocessing & Augmentation (`data_preprocessing.py`) $\rightarrow$ Training & Evaluation (`model_training.py`) $\rightarrow$ Real-time Inference HUD (`realtime_recognition.py`) $\rightarrow$ Unified CLI (`main.py`)—is functional but exhibits architectural coupling, data schema inconsistencies, and path fragility.
2. **Data Pipeline Bug & Schema Fragility**: Raw data JSON files exist in two incompatible structures: legacy 2D list `[samples, 21 landmarks]` and multi-hand 3D list `[samples, num_hands, 21 landmarks]`. In `data_preprocessing.py`, when multi-hand data is detected, single-hand files (`len(sample) == 21`) are **silently dropped** from the dataset without warning.
3. **Temporal Prediction State (R3)**: Both collection and model training operate purely on **static single-frame landmarks** ($1 \times 126$ feature vector). The scaffolded LSTM model in `model_training.py` trivially reshapes static 126-dim frames to `(1, 126)`, which provides no true temporal sequence modeling. Smoothing in real-time inference is limited to a simple majority-vote deque (`maxlen=10`, $60\%$ threshold).
4. **UI Coupling & Presentation (R1 & R2)**: Real-time recognition and UI rendering are combined in a monolithic 601-line file (`realtime_recognition.py`). While styled with cyan/amber HUD components, the drawing logic is tightly bound to inference and uses hardcoded pixel offsets.
5. **Ponytail Compliance & Bloat (R4)**: `pandas` is declared in `requirements.txt` and imported in `data_preprocessing.py` but is **never used**. `seaborn` is imported solely for `sns.heatmap` in confusion matrix plotting. `tqdm` is redundantly imported in `data_collection.py`. Path fallbacks (`'../data/raw'` vs `'data/raw'`) are inconsistently implemented.

---

## 2. Project Architecture & CLI Entry Point

### 2.1 File & Directory Map

```
sign_language_ml/
├── data/
│   ├── raw/                           # Raw JSON landmark recordings (7 files present)
│   └── processed/                     # Compressed NPZ split dataset & class_names.json
├── models/                            # Trained artifacts (.h5, SavedModel, metadata, plots)
│   ├── best_model.h5                  # Keras ModelCheckpoint weights
│   ├── confusion_matrix.png           # Test set confusion matrix
│   ├── gesture_recognition_dense_model/ # SavedModel format export
│   ├── model_metadata.json            # Model config (input_shape: 126, classes: 4)
│   └── training_history.png           # Accuracy and loss curves
├── notebooks/                         # Jupyter notebook exploration
│   ├── README.md                      # (Empty placeholder)
│   └── data_exploration.ipynb         # EDA notebook
├── src/
│   ├── camera_test.py                 # Standalone webcam diagnostic (10s feed test)
│   ├── data_collection.py             # DataCollector class (Webcam -> MediaPipe -> JSON)
│   ├── data_preprocessing.py          # GestureDataProcessor class (Norm, Aug, Split, NPZ)
│   ├── main.py                        # Unified CLI interface (argparse subparsers)
│   ├── model_training.py              # GestureModelTrainer class (Dense / LSTM, eval)
│   └── realtime_recognition.py        # RealtimeGestureRecognizer & OpenCV HUD overlay
├── generate_dummy_data.py             # Script to generate synthetic single-hand JSON data
├── INSTRUCTIONS.md                    # Git & repository publishing instructions
├── MODULES_REFERENCE.md               # API reference and documentation
├── README.md                          # Project documentation & usage instructions
├── REPOSITORY_INFO.md                 # GitHub metadata and topics
└── requirements.txt                   # Pinned project dependencies
```

### 2.2 CLI Entry Point (`src/main.py`)

`src/main.py` provides a unified command-line interface using `argparse.ArgumentParser` with subparsers for five actions:

| Subcommand | Arguments | Default Values | Target Module & Class |
|---|---|---|---|
| `collect` | `--gestures` (req, list)<br>`--samples` (int)<br>`--output` (str) | `--samples 50`<br>`--output ../data/raw` | `data_collection.DataCollector`<br>`collect_multiple_gestures()` |
| `preprocess` | `--augment` (flag)<br>`--input` (str)<br>`--output` (str) | `--augment False`<br>`--input ../data/raw`<br>`--output ../data/processed` | `data_preprocessing.GestureDataProcessor`<br>`prepare_dataset()` |
| `train` | `--model-type` (dense/lstm)<br>`--epochs` (int)<br>`--batch-size` (int)<br>`--data` (str)<br>`--output` (str) | `--model-type dense`<br>`--epochs 50`<br>`--batch-size 32`<br>`--data ../data/processed`<br>`--output ../models` | `model_training.GestureModelTrainer`<br>`train()`, `evaluate()` |
| `evaluate` | `--model` (str)<br>`--data` (str) | `--model None`<br>`--data ../data/processed` | `model_training.GestureModelTrainer`<br>`load_model()`, `evaluate()` |
| `recognize` | `--model` (str)<br>`--camera` (int)<br>`--threshold` (float)<br>`--no-flip` (flag) | `--model None`<br>`--camera 0`<br>`--threshold 0.7`<br>`--no-flip False` | `realtime_recognition.RealtimeGestureRecognizer`<br>`run()` |

### 2.3 CLI Observations & Path Resolution Fragility

1. **Relative Path Discrepancy**:
   - Default argument values in `main.py` (lines 30, 37, 39, 50, 52) default to parent-relative paths (`../data/raw`, `../data/processed`, `../models`).
   - If executed from the project root (`python src/main.py collect ...`), standard `argparse` passes `../data/raw` which resolves to `d:\Development Project\data\raw` (outside the project root!).
   - In `data_preprocessing.py` (lines 37-45) and `model_training.py` (lines 42-45), hardcoded heuristics exist:
     ```python
     if data_dir == '../data/raw' and os.path.exists('data/raw'):
         self.data_dir = 'data/raw'
     ```
     However, this fallback is **missing** in `data_collection.py` (lines 46-47 creates `../data/raw` directly) and in `realtime_recognition.py` (line 588 hardcodes `models_dir = '../models'`).
2. **Lazy Subcommand Imports**:
   - `main.py` imports modules inside subcommand handler blocks (e.g. line 83 `from data_collection import DataCollector`). When invoked as `python src/main.py`, Python sets `sys.path[0] = '.../src'`, so intra-directory imports work. If imported from root or via tests, imports without `src.` will fail unless `src` is in `PYTHONPATH`.
3. **Error Handling in `main.py`**:
   - Handlers for `train` and `evaluate` catch `FileNotFoundError` explicitly and return cleanly.
   - Handler for `recognize` catches generic `Exception` and logs error message.
   - Handlers for `collect` and `preprocess` have no top-level exception wrapping; internal exceptions bubble to the top.
   - Top-level `KeyboardInterrupt` is caught on lines 189–193 with a clean exit code 0.

---

## 3. Data Collection & Preprocessing Pipeline

### 3.1 Data Collection (`src/data_collection.py`)

- **Class**: `DataCollector`
- **Webcam Capture Loop**:
  - Sets camera resolution to $1280 \times 720$ (lines 90-91).
  - Flips frame horizontally (`cv2.flip(frame, 1)`) for intuitive user mirroring.
  - Converts BGR to RGB and passes to `mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)`.
  - On `SPACE` key press, begins collection; captures landmarks at intervals determined by `capture_delay` (default $0.2\text{s}$, giving $\sim 5\text{ frames/sec}$).
- **Serialized Landmark Structure**:
  - For each detected hand, extracts 21 keypoints: `[{'x': float, 'y': float, 'z': float, 'visibility': float}, ...]`.
  - Saves file as JSON:
    ```json
    {
      "gesture_name": "hello",
      "timestamp": "20260720_103025",
      "num_samples": 50,
      "landmarks": [
        [ [ {"x": ..., "y": ..., "z": ..., "visibility": ...} x 21 ] ]
      ],
      "two_hands": true
    }
    ```
- **Unused Import**: Line 15 imports `from tqdm import tqdm`, but `tqdm` is never referenced in `data_collection.py`.

### 3.2 Preprocessing & Feature Engineering (`src/data_preprocessing.py`)

- **Class**: `GestureDataProcessor`
- **Normalization Algorithm (`normalize_landmarks`)**:
  - Converts 21 landmark dictionaries into a $(21, 3)$ numpy array of $[x, y, z]$.
  - Identifies **Wrist** (index 0) and **Middle Finger MCP** (index 9).
  - Calculates palm center: $\text{palm\_center} = \frac{\text{wrist} + \text{middle\_mcp}}{2}$.
  - Translates points: $\text{centered\_points} = \text{points} - \text{palm\_center}$.
  - Computes scale reference: $s = \|\text{middle\_mcp} - \text{wrist}\|_2$.
  - Normalizes coordinates: $\text{normalized\_points} = \frac{\text{centered\_points}}{s}$ (if $s > 0$).
  - **Properties**: Invariant to translation across the frame and invariant to distance from camera (hand scale). Orientation/rotation is preserved.
- **Flattening & Multi-Hand Padding (`flatten_landmarks`, `prepare_dataset`)**:
  - Single hand: $21 \times 3 = 63$ features.
  - Two hands: $2 \times 63 = 126$ features.
  - If a two-handed model is used but only 1 hand is present in a frame: 63 normalized coordinates are concatenated with 63 zeros (`np.concatenate([flattened, np.zeros(63)])`).
- **Data Augmentation (`augment_landmarks`)**:
  - Generates $N$ augmented variants (default $N=5$):
    - 2D rotation around Z-axis by $\theta \sim \mathcal{U}(-0.2, 0.2)$ rad.
    - 3D translation offset $\Delta \sim \mathcal{U}(-0.1, 0.1)^3$.
    - Scaling perturbation $k \sim \mathcal{U}(0.9, 1.1)$.
- **Dataset Splitting & Export (`save_processed_data`, `load_processed_data`)**:
  - Splits data: Train ($70\%$), Validation ($10\%$), Test ($20\%$) using `train_test_split` with stratification on labels (`stratify=y`).
  - Saves compressed archive `data/processed/processed_gesture_data.npz` containing arrays `X_train, y_train, X_val, y_val, X_test, y_test, class_names, feature_dim, num_classes, is_two_handed`.
  - Also saves `data/processed/class_names.json`.

### 3.3 Critical Data Pipeline Bug: Silent Sample Loss

In `src/data_preprocessing.py` (lines 84–86 and lines 239–288):
1. `load_gesture_data()` inspects all JSON files. If **any** file contains `'two_hands': True`, `is_two_handed` is set to `True` for the entire dataset.
2. In `prepare_dataset()`:
   ```python
   if is_two_handed:
       if len(sample) == 1:
           # Hand 1 normalized + 63 zeros
       elif len(sample) == 2:
           # Hand 1 normalized + Hand 2 normalized
   ```
3. In legacy / dummy data files (e.g., `hello_20260720_101940.json` generated by `generate_dummy_data.py`), `sample` is a flat list of 21 landmark dictionaries. Therefore, `len(sample) == 21`.
4. When `is_two_handed == True`, `len(sample) == 21` satisfies neither `len(sample) == 1` nor `len(sample) == 2`.
5. **Impact**: All 150 samples across `hello_20260720_101940.json`, `no_20260720_101940.json`, and `yes_20260720_101940.json` are **silently discarded** during preprocessing without raising an error or logging a warning.
6. **Solution**: Normalize sample structure at load time (ensure `sample` is always represented as `List[HandLandmarks]` regardless of legacy or current recording format).

---

## 4. Model Training & Evaluation (`src/model_training.py`)

### 4.1 Dense Neural Network Architecture

- **Class**: `GestureModelTrainer`
- **Dense Architecture (`build_model`)**:
  - `Input(shape=(126,))`
  - `Dense(128, activation='relu')` $\rightarrow$ `BatchNormalization()` $\rightarrow$ `Dropout(0.3)`
  - `Dense(64, activation='relu')` $\rightarrow$ `BatchNormalization()` $\rightarrow$ `Dropout(0.3)`
  - `Dense(num_classes, activation='softmax')`
  - Optimizer: `Adam(learning_rate=0.001)`
  - Loss: `sparse_categorical_crossentropy`
  - Total Parameters: $\approx 25,600$ parameters (lightweight, $<0.1\text{ms}$ CPU inference).

### 4.2 Scaffolded LSTM Model (Analysis & Gaps)

- **LSTM Architecture (`build_lstm_model`)**:
  - Input: `shape=(input_features,)` (e.g. 126).
  - Reshape layer: `layers.Reshape((1, 126), input_shape=(126,))` (lines 122-127).
  - `LSTM(128, return_sequences=True)` $\rightarrow$ `Dropout(0.3)`
  - `LSTM(64)` $\rightarrow$ `Dropout(0.3)`
  - `Dense(64, activation='relu')` $\rightarrow$ `BatchNormalization()` $\rightarrow$ `Dropout(0.3)`
  - `Dense(num_classes, activation='softmax')`
- **Deficiency**:
  - The model feeds a sequence length of **1** into two recurrent LSTM layers.
  - An LSTM with $T=1$ is mathematically degenerate: it receives zero temporal transition information, computes only a non-linear projection equivalent to standard feedforward units with recurrent overhead, and wastes compute.
  - To genuinely benefit from LSTM or temporal modeling for dynamic signs, data collection must capture continuous sequences of $T \in [15, 30]$ frames (or a temporal sliding window must be formed during preprocessing/inference).

### 4.3 Training Callbacks & Artifacts

- Callbacks:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)`
  - `ModelCheckpoint(filepath='.../best_model.h5', monitor='val_accuracy', save_best_only=True)`
- Evaluation & Plots:
  - `evaluate(X_test, y_test)` generates classification report, confusion matrix plot (`confusion_matrix.png`), and training curves (`training_history.png`).
- Export Formats:
  - Saves Keras SavedModel directory (`models/gesture_recognition_dense_model/`).
  - Saves `models/model_metadata.json` (`model_type`, `class_names`, `input_shape`, `num_classes`, `timestamp`, `is_two_handed`).

---

## 5. Real-Time Recognition & UI Overlay (`src/realtime_recognition.py`)

### 5.1 Architecture & Recognition Loop

- **Class**: `RealtimeGestureRecognizer`
- **Inference Pipeline**:
  1. Capture webcam frame $\rightarrow$ flip horizontally (`cv2.flip(image, 1)`).
  2. MediaPipe `Hands.process(rgb)` extracts landmarks and `multi_handedness`.
  3. Landmark feature extraction:
     - If both hands detected: extracts left and right hand features into 126-dim vector.
     - If single hand detected on two-hand model: zero-pads 63 features to 126 dims.
  4. Model inference: `trainer.predict(features)` yields raw `(predicted_class, confidence)`.
  5. Temporal smoothing:
     - Predictions pushed to `history_buffer = deque(maxlen=10)`.
     - `get_smoothed_prediction()` computes modal class across the buffer. If modal frequency $\ge 60\%$, returns smoothed prediction; otherwise returns `"UNCERTAIN"`.
  6. Gesture sequence tracker:
     - Appends confirmed gesture to `sequence_buffer` if different from previous gesture.
     - Resets sequence if gap between gestures exceeds `sequence_timeout = 2.0s`.

### 5.2 UI Layout & Visual Presentation

The UI draws directly onto the OpenCV frame using layered blending:
- **Top Bar** (`draw_top_bar`): Dark translucent banner (`alpha=0.80`) displaying title (`"Sign Language AI"`) in cyan (`(200, 220, 0)`) and rolling average FPS in green (`(0, 220, 100)`).
- **Hand Skeleton** (`draw_hand_skeleton`): Custom colored bone connections (teal/blue) and joint vertices with accentuated fingertip circles and wrist marker.
- **Gesture Legend** (`draw_gesture_legend`): Semi-transparent sidebar in top-right displaying all recognizable classes.
- **Detection Panel** (`draw_detection_panel`): Bottom card with status indicator dot, pulsing animated border (`COL_PULSE_A` / `COL_PULSE_B`), recognized gesture title, and confidence progress bar (`draw_confidence_bar`).
- **Sequence Panel** (`draw_sequence_panel`): Sub-panel displaying chronological detected gesture flow separated by arrows (`HELLO > YES > THANKS`).
- **Controls Bar** (`draw_controls_bar`): Bottom keybind reference: `[Q] Quit`, `[C] Clear`, `[S] Screenshot`.

### 5.3 Modularity Violations & Left/Right Disambiguation Issues

1. **Tight Coupling (R2 Violation)**:
   - `realtime_recognition.py` (601 lines) embeds camera capture, MediaPipe inference, NumPy preprocessing, smoothing deques, state tracking, and 8 separate OpenCV rendering functions into one class.
   - Clean separation requires extracting the UI rendering into a dedicated view / renderer component and keeping `RealtimeGestureRecognizer` focused purely on tracking and prediction.
2. **Left vs Right Hand Ambiguity**:
   - MediaPipe classifies handedness in the camera's original (unflipped) coordinate system.
   - In lines 465–467, the image is flipped horizontally with `cv2.flip(image, 1)`. MediaPipe processes this flipped image. MediaPipe's neural network expects a non-mirrored viewpoint; when run on a mirrored frame, it frequently inverts `"Left"` and `"Right"` classifications.
   - In lines 516–519, if handedness labels are missing or identical, it assigns the first detected hand to index 0..62 (left) and second to 63..125 (right) based purely on detection order rather than spatial position (x-coordinate), causing feature swaps when hands cross.

---

## 6. Camera Initialization Check (`src/camera_test.py`)

- **Implementation**:
  - Queries `cv2.VideoCapture(0)`.
  - Verifies device opening (`cap.isOpened()`).
  - Displays feed in an OpenCV window for 10 seconds or until `'q'` is pressed.
  - Releases camera and destroys windows on exit.
- **Assessment**:
  - Robust, simple diagnostic conforming to Ponytail's minimal runnable check principle.
  - Minor enhancement opportunity: Accept an optional camera index CLI argument to mirror `main.py --camera`.

---

## 7. Ponytail Compliance & Lazy Senior Dev Analysis

Applying the principles from `.agents/Ponytail skills/AGENTS.md` (deletion over addition, YAGNI, standard library over external deps, minimal working diff):

### 7.1 Dead Code & Unused Dependencies

| Item | Location | Current State | Ponytail Action |
|---|---|---|---|
| `pandas` | `requirements.txt:2`<br>`src/data_preprocessing.py:12` | Imported as `import pandas as pd`, never called anywhere in the codebase. | Remove from `requirements.txt` and `src/data_preprocessing.py`. |
| `seaborn` | `requirements.txt:4`<br>`src/model_training.py:16` | Used only for `sns.heatmap` in confusion matrix plotting. | Replace with native `matplotlib.pyplot.imshow` or `matshow`; remove `seaborn` dependency. |
| `tqdm` (in collector) | `src/data_collection.py:15` | Imported but never used in `data_collection.py`. | Remove unused import. |
| `jupyter`, `ipykernel` | `requirements.txt:10-11` | Notebook runtime packages pinned in core production requirements. | Prune from minimal deployment requirements. |
| `extract_landmarks_from_file` | `MODULES_REFERENCE.md:43` | Documented in reference docs but never implemented in `data_collection.py`. | Reconcile documentation with code. |
| Pseudo-LSTM `(1, 126)` | `src/model_training.py:107-156` | Ineffectual sequence length 1 LSTM. | Replace with genuine temporal smoothing / temporal windowing or clean up. |

### 7.2 Code Duplication & Inconsistencies

1. **Path Normalization**: Constructors in multiple files implement duplicate ad-hoc path checks (`if data_dir == '../data/raw' and os.path.exists('data/raw')`). Standardizing all default paths to project-root relative paths (`data/raw`, `data/processed`, `models`) cleans up ~30 lines of boilerplate.
2. **Double Persistence**: `data_preprocessing.py` saves `class_names` both inside `processed_gesture_data.npz` and in a separate `class_names.json`.
3. **Hardcoded UI Magic Numbers**: Panel widths, heights, colors, and font scales are hardcoded across 8 drawing functions in `realtime_recognition.py`.

---

## 8. Enumerated Functional Gaps & Requirements Traceability

| ID | Requirement Area | Current Code State | Functional Gap / Defect | Target Phase |
|---|---|---|---|---|
| **G1** | **R1. UI Presentation** | Drawing functions intertwined with camera loop in `realtime_recognition.py`. | No separation of UI renderer; hardcoded pixel coordinates; lack of responsive layout for varying camera aspect ratios (4:3 vs 16:9). | Phase 1 & 3 |
| **G2** | **R2. Modularity** | Single monolithic `realtime_recognition.py` file (601 lines). | Recognition loop, gesture state machine, landmark feature normalization, and OpenCV graphics are coupled. | Phase 1 |
| **G3** | **R3. Temporal Prediction** | Static frame classifier + simple majority-vote deque (`maxlen=10`, 60% threshold). | Cannot capture dynamic motion signatures; naive deque causes jitter/lag during gesture transitions; LSTM in `model_training.py` only takes sequence length 1. | Phase 2 |
| **G4** | **Data Pipeline Schema** | `load_gesture_data` sets global `is_two_handed=True` if one file has 2 hands; drops `len(sample)==21`. | Incompatibility between single-hand recordings and multi-hand recordings; silent data loss of legacy/synthetic samples. | Phase 1 |
| **G5** | **Hand Disambiguation** | Arbitrary index assignment when handedness confidence is low. | Mirrored video feed causes left/right hand flipping; crossing hands swaps left and right 63-dim feature vectors. | Phase 1 & 2 |
| **G6** | **R4. Code Cleanliness** | Unused `pandas`, `seaborn`, `tqdm` imports; path resolution inconsistencies. | Violates Ponytail guidelines: extra dependencies, bloated imports, and inconsistent relative path resolution. | Phase 1 |
| **G7** | **Environment & NumPy 2.x** | System Python with NumPy 2.x causes `ImportError: cannot import name 'ComplexWarning' from 'numpy.core.numeric'`. | Project requires running inside the configured virtual environment (`venv/Scripts/python.exe`) or updating `requirements.txt` to compatible modern package matrix. | Phase 0/1 |

---

## 9. Conclusion & Recommendations

The Sign Language Detection ML project has a solid foundational computer vision workflow and clean model performance on static poses. To satisfy the prompt objectives under Ponytail senior developer principles:
1. **Phase 1 (Modularity & Cleanup)**: Decouple `realtime_recognition.py` into a clean `SignLanguageRecognizer` engine and a dedicated `HUDOverlay` UI renderer. Fix the data schema parsing bug in `data_preprocessing.py`. Eliminate unused imports (`pandas`, `seaborn`, `tqdm`). Standardize CLI default paths to project root.
2. **Phase 2 (Temporal Prediction)**: Implement an Exponential Moving Average (EMA) probability filter with adaptive confidence hysteresis or upgrade to a sliding-window temporal feature extractor to eliminate prediction flicker and provide rock-solid temporal stability.
3. **Phase 3 (UI Polish & Verification)**: Refine the HUD overlay with dynamic responsive anchoring, smooth progress animations, clean skeleton rendering, and execute end-to-end verification across `camera_test.py`, `preprocess`, `train`, and `main.py --help`.
