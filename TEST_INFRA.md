# Test Infrastructure & Specification

## 1. Overview & Architecture Philosophy

The Sign Language Detection ML System test infrastructure is engineered according to **Ponytail Senior Developer Guidelines**:
- **Zero New External Framework Dependencies**: Built 100% on Python standard library `unittest` and standard math/imaging primitives (`numpy`, `cv2`, `tensorflow`).
- **High-Velocity Execution**: Fast feedback loops categorized into 5 distinct execution tiers with strict SLA budgets.
- **Hermetic Isolation**: Tests run deterministically without hardware requirements (no physical webcam required, headless display-safe), creating isolated temporary directories and in-memory mock structures.
- **Behavioral & Invariant Verification**: Verifies core mathematical invariants (coordinate scale/translation invariance, probability conservation, hysteresis state stability) and defensive boundary behavior rather than superficial implementation details.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TEST RUNNER                                   │
│                     (tests/run_tests.py)                                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  ┌──────────────┐            ┌──────────────┐            ┌──────────────┐
  │    TIER 1    │            │    TIER 2    │            │    TIER 3    │
  │ Invariants & │            │ Algorithmic  │            │ Mock-Driven  │
  │ Fast Schema  │            │ & ML Models  │            │ UI & Camera  │
  │  (< 50ms)    │            │  (< 200ms)   │            │  (< 500ms)   │
  └──────────────┘            └──────────────┘            └──────────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  ┌──────────────┐            ┌──────────────┐
  │    TIER 4    │            │    TIER 5    │
  │ End-to-End   │            │ Adversarial  │
  │ Pipelines    │            │ & Stress     │
  │  (< 2000ms)  │            │  (< 1000ms)  │
  └──────────────┘            └──────────────┘
```

---

## 2. 5-Tier Testing Architecture

| Tier | Focus Area | Modules Tested | Target SLA | Execution Frequency |
|---|---|---|---|---|
| **Tier 1: Fast Invariants & Schema** | Normalization math, schema parsing, dimension validation, data contracts | `data_preprocessing.py`, `ui_overlay.py` | < 50ms | Every commit / pre-commit |
| **Tier 2: Algorithmic & ML** | Softmax EMA, dual hysteresis, velocity gating, direct tensor inference | `model_training.py`, `realtime_recognition.py` | < 200ms | Every PR / commit |
| **Tier 3: Mock-Driven UI & HW** | Headless HUD rendering, ROI blending, camera diagnostics, CLI parsing | `camera_test.py`, `realtime_recognition.py`, `main.py` | < 500ms | Pre-merge integration |
| **Tier 4: Pipeline E2E** | Full data cycle (JSON -> NPZ -> Model Train -> Predict) | Entire `src/` pipeline | < 2000ms | CI nightly / release build |
| **Tier 5: Adversarial & Stress** | Corrupted inputs, zero divisions, NaN landmarks, disconnect simulation | All modules | < 1000ms | Stress testing / audits |

---

## 3. Test Suite Directory Layout

```
d:\Development Project\Sign Language\
├── TEST_INFRA.md                # Test infrastructure & architectural contract
├── TEST_READY.md                # Test suite status, verification instructions, test inventory
├── tests/
│   ├── __init__.py              # Test package initializer & environment path setup
│   ├── run_tests.py             # Unified tier-by-tier test runner CLI
│   ├── test_cli.py              # CLI argument parser, help output, and subcommands
│   ├── test_preprocessing.py    # Schema parser, normalization invariants, augmentation
│   ├── test_model.py            # Dense MLP architecture, direct tensor inference, export/import
│   ├── test_temporal.py         # Softmax EMA, hysteresis debouncing, velocity gate, sequences
│   ├── test_ui.py               # HUD rendering without camera, ROI alpha-blending, HUDState
│   └── test_camera.py           # Camera diagnostic initialization, mock frames, error handling
```

---

## 4. Invariant Contracts & Mathematical Properties

### 4.1. Landmark Normalization Invariants
- **Translation Invariance**: For any landmark set $L$ and constant translation vector $\vec{t} \in \mathbb{R}^3$:
  $$\text{Normalize}(L + \vec{t}) \equiv \text{Normalize}(L)$$
- **Scale Invariance**: For any landmark set $L$ and scalar scale $s > 0$:
  $$\text{Normalize}(s \cdot L) \equiv \text{Normalize}(L)$$
- **Origin Centering**: The palm center $\frac{\vec{p}_{\text{wrist}} + \vec{p}_{\text{middle\_mcp}}}{2}$ is mapped strictly to $(0, 0, 0)$.
- **Unit Reference Distance**: The Euclidean distance $\|\vec{p}_{\text{middle\_mcp}} - \vec{p}_{\text{wrist}}\| = 1.0$ (when reference distance $> 0$).
- **Zero-Division Defense**: Degenerate hands with zero wrist-to-MCP distance remain centered without raising `ZeroDivisionError` or producing `NaN`/`Inf`.

### 4.2. Model Inference Contract
- **Direct Callable Execution**: Real-time evaluation executes directly via `model(x, training=False).numpy()` (~1ms execution) rather than `model.predict(x)` (which introduces graph tracing and allocator overhead).
- **Probability Conservation**: For every inference vector $\vec{p}$, $\sum_{i=1}^C p_i = 1.0$ and $p_i \ge 0 \quad \forall i$.
- **Tensor Dimensionality**: Single-hand input shape is `(None, 63)`; two-handed input shape is `(None, 126)`.

### 4.3. Temporal Filter & Debouncing Invariants
- **Softmax Exponential Moving Average (EMA)**:
  $$S_t = \alpha P_t + (1 - \alpha) S_{t-1}, \quad \alpha = 0.25$$
  - Preserves sum to 1.0 across all classes.
  - Smooths momentary spikes and eliminates single-frame jitter.
- **Dual-Threshold Hysteresis**:
  - Activation: Transitions from `SCANNING` $\to$ `DETECTED` if and only if $P(\text{class}) \ge T_{\text{high}} = 0.80$ sustained for $N \ge 4$ frames.
  - Retention: Remains in `DETECTED` state while $P(\text{class}) \ge T_{\text{low}} = 0.45$.
  - Deactivation: Transitions back to `SCANNING` / `UNCERTAIN` when $P(\text{class}) < T_{\text{low}} = 0.45$.
- **Kinematic Velocity Gating**: Suppresses false positives during rapid transit ($v_{\text{wrist}} > v_{\text{max}}$), enabling prediction only during steady gesture holds.

### 4.4. Decoupled UI & ROI Blending Contract
- **Sub-Array ROI Blending**:
  $$I_{\text{out}}[y:y+h, x:x+w] = \alpha \cdot C_{\text{overlay}} + (1 - \alpha) \cdot I_{\text{in}}[y:y+h, x:x+w]$$
  Achieves < 0.05ms overlay latency by avoiding full-frame array duplication.
- **Headless Safety**: All UI rendering executes cleanly on in-memory NumPy frame buffers without requiring an active X11/Win32 display server.

---

## 5. Execution Commands

### 5.1. Unified Test Runner (Recommended)
```powershell
# Run the entire test suite across all 5 tiers
python tests/run_tests.py

# Run a specific tier (e.g. Tier 1 Invariants or Tier 2 Algorithmic)
python tests/run_tests.py --tier 1
python tests/run_tests.py --tier 2
python tests/run_tests.py --tier 3
python tests/run_tests.py --tier 4
python tests/run_tests.py --tier 5

# Verbose output with timing per individual test case
python tests/run_tests.py -v

# Stop immediately on first test failure
python tests/run_tests.py --failfast
```

### 5.2. Standard Python `unittest` Invocation
```powershell
# Run all tests using standard library discovery
python -m unittest discover -s tests -p "test_*.py" -v

# Run single test module
python -m unittest tests/test_preprocessing.py
python -m unittest tests/test_model.py
python -m unittest tests/test_temporal.py
python -m unittest tests/test_ui.py
python -m unittest tests/test_camera.py
python -m unittest tests/test_cli.py
```

---

## 6. Exit Codes & CI/CD Integration

- `0` : All tests passed successfully within SLA budgets.
- `1` : One or more tests failed or encountered an assertion error.
- `2` : Test environment / configuration error (missing required core dependencies).
