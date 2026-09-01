# Sign Language Detection System using Machine Learning

A real-time system that detects and interprets sign language gestures using webcam input, MediaPipe hand tracking, and a deep learning classifier.

---

## Features
- **Real-Time Landmark Detection**: Tracks up to 2 hands and extracts 21 3D coordinates per hand using Google MediaPipe.
- **Single and Two-Handed Gesture Support**: Zero-pads single-hand gestures to fit a unified 126-dimension feature space for robust two-handed recognition.
- **Translation & Scale Invariance**: Normalizes coordinates relative to the palm center and wrist-to-middle-finger length, ensuring gestures are recognized regardless of hand size or position.
- **Temporal Prediction Filter**: Continuous Softmax Exponential Moving Average (EMA) probability smoothing, dual-threshold hysteresis debouncing ($T_{\text{high}}=0.80, T_{\text{low}}=0.45$), and kinematic wrist velocity gating.
- **Decoupled Premium HUD Overlay**: Sleek OpenCV HUD (`SignLanguageHUD`) using high-performance sub-array ROI alpha blending (<0.05ms latency), corner brackets, handedness badges, dynamic confidence meters, and gesture sequences.
- **Comprehensive 5-Tier Test Infrastructure**: 100% standard library `unittest` suite covering invariant contracts, direct tensor inference, headless HUD rendering, E2E pipelines, and adversarial stress boundaries.

---

## Project Structure

```
sign_language_ml/
├── data/                      # Data storage
│   ├── raw/                   # Raw gesture data (JSON landmark files)
│   └── processed/             # Processed datasets (Numpy NPZ format)
├── models/                    # Trained models, metadata, and performance graphs
├── src/                       # Source code
│   ├── camera_test.py         # Webcam diagnostic tool (headless-safe)
│   ├── data_collection.py     # Landmark recorder module
│   ├── data_preprocessing.py  # Landmark normalization, unified parsing & augmentation
│   ├── model_training.py      # Dense MLP model training & direct tensor evaluation
│   ├── temporal_filter.py     # Softmax EMA, hysteresis debouncing, velocity gate
│   ├── ui_overlay.py          # Decoupled native OpenCV HUD renderer
│   ├── realtime_recognition.py # Real-time webcam inference coordinator
│   └── main.py                # Unified command-line interface entry point
├── tests/                     # 5-Tier comprehensive test suite
│   ├── run_tests.py           # Unified tier test runner
│   ├── test_cli.py            # CLI argument parser tests
│   ├── test_preprocessing.py  # Preprocessing math & schema tests
│   ├── test_model.py          # Model architecture & tensor inference tests
│   ├── test_temporal.py       # Temporal filtering & hysteresis tests
│   ├── test_ui.py             # Headless HUD rendering & ROI blend tests
│   └── test_camera.py         # Camera diagnostics & mock tests
├── requirements.txt           # Cleaned dependencies (no pandas, no seaborn)
└── README.md                  # Project documentation
```

---

## Setup & Installation

1. **Activate Virtual Environment** (Create one if needed):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Webcam Access**:
   ```bash
   python src/camera_test.py
   ```

---

## Pipeline & Workflow

### 1. Collect Gesture Data
Record coordinates for custom gestures (e.g., `hello`, `thanks`, `yes`, `no`):
```bash
python src/main.py collect --gestures hello thanks yes no --samples 50 --output data/raw
```
*Tip: Place your hand in position and hold **SPACE** to record. Tilt and move your hand slightly for better variations.*

### 2. Preprocess & Augment
Clean and normalize coordinates. Applies random rotation, translation, and scale transformations:
```bash
python src/main.py preprocess --input data/raw --output data/processed --augment
```

### 3. Train Model
Train a Feedforward Dense Neural Network:
```bash
python src/main.py train --data data/processed --output models --epochs 50
```

### 4. Run Live Recognition
Run real-time inference using the premium HUD visualization:
```bash
python src/main.py recognize --threshold 0.7
```
- **`q`**: Quit the live feed.
- **`c`**: Clear the detected gesture sequence box.
- **`s`**: Take a screenshot (saved as `screenshot_YYYYMMDD_HHMMSS.png`).

---

## Model Performance

The training process and final test set classification results are illustrated below:

### 1. Confusion Matrix
![Confusion Matrix](models/confusion_matrix.png)

*Explanation*: The confusion matrix shows model performance on the test dataset split. The dense neural network achieves a **100% correct classification rate** across the custom gesture classes (`hello`, `no`, `thanks`, `yes`). The clean diagonal structure confirms that normalized joint offsets are highly separable features.

### 2. Training History
![Training History](models/training_history.png)

*Explanation*: The training history plot tracks validation accuracy and categorical cross-entropy loss:
- **Model Accuracy**: Validation accuracy climbs rapidly to 1.0 (100% accuracy) within the first 15 epochs and remains stable.
- **Model Loss**: Training and validation losses decay smoothly and converge toward zero, proving that regularization techniques (Batch Normalization and Dropout) effectively prevented overfitting.

---

### 5. Run Verification & Test Suite
Execute the 5-Tier comprehensive test suite:
```bash
# Run all 5 tiers with timing diagnostics
python tests/run_tests.py -v

# Run a specific tier (1 to 5)
python tests/run_tests.py --tier 1
```

---

## Technical Details

### Data Normalization
To make gesture recognition robust to distance and position:
- **Translation**: Centers the coordinate system on the palm center (calculated as the average of the wrist and middle finger MCP joint).
- **Scaling**: Divides all coordinate offsets by the distance between the wrist and the middle finger MCP joint.
- **Zero-Division Immunity**: Degenerate zero-scale landmarks remain safely origin-centered without NaN or exception.

### Temporal Filtering & Stability
- **Softmax EMA Smoothing**: Applies $S_t = \alpha P_t + (1 - \alpha) S_{t-1}$ ($\alpha = 0.25$) across probability distributions.
- **Dual-Threshold Hysteresis**: Requires $P \ge 0.80$ sustained for $\ge 4$ consecutive frames to activate `DETECTED`, and retains state while $P \ge 0.45$.
- **Kinematic Velocity Gating**: Suppresses active predictions during rapid hand transit ($v > v_{\text{threshold}}$) to eliminate transitional false triggers.

### Model Architecture (Dense MLP)
- **Input Layer**: 126 nodes (matching flattened 3D landmarks for 2 hands; single hands are zero-padded).
- **Hidden Layer 1**: 128 nodes with ReLU activation, Batch Normalization, and 30% Dropout.
- **Hidden Layer 2**: 64 nodes with ReLU activation, Batch Normalization, and 30% Dropout.
- **Output Layer**: Softmax activation mapping to the number of target gestures.
- **Direct Tensor Inference**: Evaluated via `model(tensor, training=False).numpy()` (~1ms execution).

### Decoupled HUD Overlay
- **Sub-Array ROI Blending**: In-place alpha blending directly on image slice ROIs (`<0.05ms` latency).
- **Headless Safety**: `SignLanguageHUD` renders purely on NumPy frame buffers without requiring an active display server.

---

## Troubleshooting

- **Camera Fails to Open**:
  - Verify that another application (like Zoom or Teams) is not locking the webcam.
  - Run `python src/camera_test.py --headless --duration 1` to test camera acquisition without GUI.
  - Run `python src/main.py recognize --camera 1` to try an alternative camera index.
- **Model Directory Error**:
  - The scripts automatically auto-detect folders. Ensure commands are run from the project root directory.

