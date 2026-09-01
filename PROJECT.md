# Project: Sign Language Detection ML System Refactoring

## Architecture
The Sign Language Detection ML system captures video frames via OpenCV, extracts 21 hand landmarks per hand via MediaPipe Hands, normalizes landmarks to wrist/palm coordinate frames, performs ML inference with a TensorFlow/Keras Dense Neural Network, applies temporal probability smoothing with hysteresis and velocity gating, and renders a native OpenCV HUD overlay displaying predictions, confidence metrics, gesture sequences, and diagnostic information.

```
[Camera / Video Stream]
         │
         ▼
[MediaPipe Hands Landmark Extraction] (21 landmarks x 2 hands)
         │
         ▼
[Landmark Normalization] (Center to palm, scale to hand size)
         │
         ▼
[TensorFlow Dense MLP Inference] (Direct tensor evaluation: ~1ms)
         │
         ▼
[Temporal Prediction Filter] (Softmax EMA + Dual-Threshold Hysteresis + Velocity Gating)
         │
         ▼
[HUD State Container] (Decoupled state representation)
         │
         ▼
[SignLanguageHUD Renderer] (High-performance ROI-blended OpenCV HUD)
```

## Feature Inventory
Every feature identified during the survey phase is mapped to a designated milestone below:

| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Standardized Root Paths | Fix CLI and default paths in `main.py`, `data_collection.py`, `data_preprocessing.py`, and `realtime_recognition.py` to resolve relative to project root. | M1 | Survey (Explorer 1) | DONE |
| 2 | Unified Raw Data Parser | Fix `data_preprocessing.py` to parse both legacy single-hand 2D/3D flat formats (`len==21`) and multi-hand formats (`len in (1,2)`) without silent data loss. | M1 | Survey (Explorer 1) | DONE |
| 3 | Ponytail Dependency Pruning | Remove unused dependencies (`pandas`, `seaborn`) and dead imports (`tqdm` in `data_collection.py`), streamline `requirements.txt`. | M1 | Survey (Explorer 1) | DONE |
| 4 | Camera Initialization Test | Streamline and harden `src/camera_test.py` for headless safety, timeout protection, and device index handling. | M1 | Survey (Explorer 1) | DONE |
| 5 | Model Training Streamlining | Clean up `src/model_training.py`, prune/clarify pseudo-LSTM, optimize Dense MLP training with native matplotlib confusion matrix (no seaborn). | M2 | Survey (Explorer 2) | DONE |
| 6 | Softmax Probability EMA | Implement continuous Exponential Moving Average over class probability distributions in `realtime_recognition.py`. | M2 | Survey (Explorer 2) | DONE |
| 7 | Dual-Threshold Hysteresis | Implement debounced gesture state transitions ($T_{\text{high}}=0.80, T_{\text{low}}=0.45$) to eliminate boundary flicker. | M2 | Survey (Explorer 2) | DONE |
| 8 | Kinematic Velocity Gating | Add wrist displacement velocity thresholding to suppress false triggers during rapid hand transitions. | M2 | Survey (Explorer 2) | DONE |
| 9 | Direct Tensor Inference | Replace slow Keras `model.predict()` calls in real-time loops with direct callable execution `model(x, training=False).numpy()`. | M2 | Survey (Explorer 2) | DONE |
| 10 | UI & Recognition Decoupling | Extract all drawing routines from `realtime_recognition.py` into a dedicated `SignLanguageHUD` class driven by a clean `HUDState` dataclass. | M3 | Survey (Explorer 3) | DONE |
| 11 | High-Performance ROI Blending | Replace full-frame copying in `_overlay_rect` with sub-array ROI blending, dropping overlay latency to <0.05ms. | M3 | Survey (Explorer 3) | DONE |
| 12 | Native OpenCV Premium HUD | Implement modern visual elements: sleek top HUD, corner-bracket hand bounding boxes, handedness badges, threshold confidence meters, sequence countdowns. | M3 | Survey (Explorer 3) | DONE |
| 13 | Comprehensive E2E Test Suite | 5-tier test suite covering CLI, pipelines, camera test, ML models, temporal filtering, UI rendering, and edge cases. | M-E2E | Dual Track | DONE |
| 14 | Final Integration & Hardening | Full end-to-end verification across all tiers with adversarial stress testing. | M4 | Implementation Track | DONE |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M-E2E | E2E Test Track | Design and build comprehensive 5-tier test suite (`tests/`) and publish `TEST_READY.md`. | none | DONE |
| M1 | Pipeline & Ponytail Cleanup | Fix CLI path resolution, unify raw data parser in `data_preprocessing.py`, prune `pandas`/`seaborn`, harden `camera_test.py`. | none | DONE |
| M2 | ML & Temporal Prediction | Upgrade temporal stability (Softmax EMA, hysteresis, velocity gate, direct tensor inference) and streamline `model_training.py`. | M1 | DONE |
| M3 | UI Decoupling & Premium HUD | Create `SignLanguageHUD` with ROI blending, sleek OpenCV widgets, and decouple `realtime_recognition.py`. | M1, M2 | DONE |
| M4 | Final Integration & E2E Pass | Run 100% of E2E tests across Tiers 1-4, execute Tier 5 adversarial coverage hardening, verify all acceptance criteria. | M-E2E, M3 | DONE |

## Interface Contracts

### 1. Raw Data Parser (`src/data_preprocessing.py`)
```python
def parse_raw_sample(sample: list) -> list[list[dict]]:
    """
    Normalizes any raw sample representation into a list of hands:
    - Single hand flat: [ {x, y, z}, ... 21 dicts ] -> [ [ {x, y, z}, ... 21 dicts ] ]
    - Multi-hand list: [ [ {x, y, z}, ... 21 dicts ], ... ] -> preserved as-is
    """
```

### 2. Temporal Smoother (`src/temporal_filter.py`)
```python
class TemporalSmoother:
    def __init__(self, alpha: float = 0.25, threshold_high: float = 0.80, threshold_low: float = 0.45, debounce_frames: int = 4, velocity_threshold: float = 0.08):
        ...
    def update(self, raw_probs: np.ndarray, wrist_pos: tuple[float, float] | None = None, dt: float = 1.0/30.0) -> tuple[str, float, bool]:
        """
        Returns (predicted_class, smoothed_confidence, is_stable_gesture)
        """
```

### 3. Decoupled HUD Interface (`src/ui_overlay.py` / `src/realtime_recognition.py`)
```python
@dataclass
class HUDState:
    fps: float = 0.0
    detected: bool = False
    status_text: str = "SCANNING"  # "SCANNING" | "DETECTED" | "UNCERTAIN"
    gesture: str = "None"
    confidence: float = 0.0
    sequence: list[str] = field(default_factory=list)
    hands_info: list[dict] = field(default_factory=list)  # list of {bbox, landmarks, handedness}
    classes: list[str] = field(default_factory=list)
    sequence_progress: float = 0.0

class SignLanguageHUD:
    def __init__(self, class_names: list[str] | None = None):
        ...
    def render(self, frame: np.ndarray, state: HUDState) -> np.ndarray:
        """Renders HUD overlay elements onto frame in-place or returns composited image."""
```

## Code Layout
```
d:\Development Project\Sign Language\
├── src/
│   ├── camera_test.py           # Camera initialization check & diagnostics
│   ├── data_collection.py       # Landmark capture from webcam
│   ├── data_preprocessing.py    # Landmark normalization & dataset generation
│   ├── model_training.py        # Dense MLP training & evaluation
│   ├── temporal_filter.py       # Softmax EMA, hysteresis, velocity gating
│   ├── realtime_recognition.py  # Live camera inference loop & pipeline coordination
│   ├── ui_overlay.py            # Decoupled native OpenCV HUD renderer
│   └── main.py                  # CLI entry point
├── tests/
│   ├── test_cli.py              # CLI argument parsing & help verification
│   ├── test_preprocessing.py    # Raw data loading, schema parsing, normalization
│   ├── test_model.py            # Model building, tensor inference, dimensions
│   ├── test_temporal.py         # Softmax EMA, hysteresis debouncing, velocity gate
│   ├── test_ui.py               # HUD rendering without camera, ROI blending perf
│   ├── test_camera.py           # Camera test mock & diagnostic validation
│   └── run_tests.py             # E2E test runner
├── data/
│   ├── raw/                     # Raw landmark JSON files
│   └── processed/               # Preprocessed NPZ files
├── models/                      # Saved models & model_metadata.json
├── requirements.txt             # Cleaned dependency manifest (no pandas, no seaborn)
└── README.md                    # Project documentation
```
