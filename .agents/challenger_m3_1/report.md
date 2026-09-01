# Adversarial Challenge Report: SignLanguageHUD & HUDState

**Milestone**: M3 (UI Decoupling & Premium OpenCV HUD)  
**Agent**: teamwork_preview_challenger instance 1  
**Target Modules**: `src/ui_overlay.py`, `src/realtime_recognition.py`, `tests/test_ui.py`  
**Overall Risk Assessment**: **HIGH**  
**Verdict**: **REQUEST_CHANGES**

---

## Executive Summary

An empirical adversarial stress test was executed against `SignLanguageHUD` and `HUDState` across four dimensions:
1. **Extreme Frame Resolutions**: Tiny ($32\times32, 100\times100$), standard ($640\times480, 1280\times720, 1920\times1080$), 4K UHD ($3840\times2160$), and non-standard aspect ratios ($2560\times400, 400\times2560$).
2. **Out-of-Bounds & Malformed Coordinates**: Negative coordinates, coordinates exceeding frame dimensions, completely out-of-frame boxes, inverted boxes, zero-area boxes, and empty/partial hand landmarks.
3. **Extreme / Malformed State Properties**: Empty/None class lists, 100+ character gesture names, 50+ item sequence buffers, extreme FPS ($0.0, 9999.9, \pm\infty, \text{NaN}$), confidence extremes ($<0.0, >1.0, \pm\infty, \text{NaN}, \text{None}$), handedness scores ($\text{NaN}, \infty$), and sequence progress extremes.
4. **Performance & ROI Blending Throughput**: Benchmarked `blend_roi` standalone latency, full-frame copy comparison, and complete multi-widget `render()` latency across resolutions.

### Key Findings Summary:
- **5 Unhandled Exception Crashes Discovered**:
  1. `landmarks: []` without precomputed `bbox` triggers `ValueError: min() arg is an empty sequence` in `draw_hands()` (`src/ui_overlay.py:249`).
  2. `score = float('nan')` in `hands_info` triggers `ValueError: cannot convert float NaN to integer` in `_draw_handedness_badge()` (`src/ui_overlay.py:147`).
  3. `score = float('inf')` triggers `OverflowError: cannot convert float infinity to integer` in `_draw_handedness_badge()` (`src/ui_overlay.py:147`).
  4. `classes = [0, 1, 2]` triggers `AttributeError: 'int' object has no attribute 'capitalize'` in `draw_gesture_legend()` (`src/ui_overlay.py:441`).
  5. `confidence = None` or `fps = None` triggers `TypeError` during formatting/type casting (`src/ui_overlay.py:174, 281`).
- **ROI Blending Performance Verified**: Sub-array ROI blending is **2.1x to 12.9x faster** than full-frame compositing. Pure ROI slice blending runs in $\approx 0.04-0.08$ms for small widgets and $\approx 0.48-1.20$ms for full-width panels. Full multi-widget HUD render runs in **5.67ms** at 640x480 (~163 FPS) and **7.62ms** at 720p (~101 FPS).
- **Test Suite Decoupling Gap**: `tests/test_ui.py` tests an embedded mock `SignLanguageHUDReference` rather than importing `SignLanguageHUD` from `src/ui_overlay.py`.

---

## 1. Concrete Vulnerability Findings & Empirically Reproduced Failures

### Finding 1 [HIGH]: `ValueError` on Empty Landmark List without BBox
- **Location**: `src/ui_overlay.py:246-250` in `draw_hands()`
- **Root Cause**: When a hand entry contains `{'landmarks': []}` or an empty sequence, `lm_list` is empty. The list comprehensions `xs = [...]` and `ys = [...]` result in empty lists `[]`. Calling `min(xs)` throws `ValueError: min() arg is an empty sequence`.
- **Reproduction**:
  ```python
  from ui_overlay import SignLanguageHUD, HUDState
  hud = SignLanguageHUD()
  frame = np.zeros((720, 1280, 3), dtype=np.uint8)
  state = HUDState(hands_info=[{'landmarks': [], 'handedness': 'Right'}])
  hud.render(frame, state)
  # ValueError: min() arg is an empty sequence
  ```
- **Blast Radius**: If MediaPipe returns a detection with empty landmark container or downstream filter drops landmarks, the entire recognition loop crashes ungracefully.
- **Recommended Mitigation**:
  In `draw_hands`:
  ```python
  lm = hand.get('landmarks')
  if lm is not None:
      lm_list = lm.landmark if hasattr(lm, 'landmark') else lm
      if not lm_list or len(lm_list) < 21:
          continue
  ```

---

### Finding 2 [HIGH]: `ValueError` & `OverflowError` on NaN/Inf Handedness Score
- **Location**: `src/ui_overlay.py:147` in `_draw_handedness_badge()`
- **Root Cause**:
  ```python
  text = f"{label} ({int(score * 100)}%)" if score is not None else label
  ```
  If `score` is `float('nan')` or `float('inf')`, Python cannot convert NaN/inf to integer (`int(float('nan') * 100)`), throwing `ValueError` or `OverflowError`.
- **Reproduction**:
  ```python
  state_nan = HUDState(hands_info=[{'bbox': (100, 100, 200, 200), 'handedness': 'Right', 'score': float('nan')}])
  hud.render(frame, state_nan)
  # ValueError: cannot convert float NaN to integer

  state_inf = HUDState(hands_info=[{'bbox': (100, 100, 200, 200), 'handedness': 'Right', 'score': float('inf')}])
  hud.render(frame, state_inf)
  # OverflowError: cannot convert float infinity to integer
  ```
- **Blast Radius**: MediaPipe or custom tracking models emitting unnormalized / undefined score values crash the HUD rendering pipeline.
- **Recommended Mitigation**:
  In `_draw_handedness_badge`:
  ```python
  import math
  is_valid_score = (score is not None and isinstance(score, (int, float)) and math.isfinite(score))
  text = f"{label} ({int(score * 100)}%)" if is_valid_score else label
  ```

---

### Finding 3 [MEDIUM]: `AttributeError` on Non-String Class Names in Legend
- **Location**: `src/ui_overlay.py:441, 444` in `draw_gesture_legend()`
- **Root Cause**:
  ```python
  cv2.putText(image, name.capitalize(), (x + pad + 14, ty), ...)
  ```
  If `classes` contains numeric labels (e.g. `[0, 1, 2]`), `name.capitalize()` fails with `AttributeError: 'int' object has no attribute 'capitalize'`.
- **Reproduction**:
  ```python
  state = HUDState(classes=[0, 1, 2])
  hud.render(frame, state)
  # AttributeError: 'int' object has no attribute 'capitalize'
  ```
- **Recommended Mitigation**: Use `str(name).capitalize()`.

---

### Finding 4 [MEDIUM]: `TypeError` on `None` Values for `confidence` or `fps`
- **Location**: `src/ui_overlay.py:174, 281`
- **Root Cause**:
  - `clamped_conf = max(0.0, min(float(confidence), 1.0))` throws `TypeError` if `confidence=None`.
  - `fps_text = f"FPS: {fps_val:.0f}"` throws `TypeError: unsupported format string passed to NoneType.__format__` if `fps=None`.
- **Reproduction**:
  ```python
  hud.render(frame, HUDState(confidence=None))
  # TypeError: float() argument must be a string or a real number, not 'NoneType'

  hud.render(frame, HUDState(fps=None))
  # TypeError: unsupported format string passed to NoneType.__format__
  ```
- **Recommended Mitigation**: Coerce `confidence = 0.0 if confidence is None else float(confidence)` and `fps_val = 0.0 if fps_val is None else float(fps_val)`.

---

### Finding 5 [MEDIUM]: Out-of-Sync E2E Test Suite (`tests/test_ui.py`)
- **Location**: `tests/test_ui.py:42-139`
- **Observation**: `tests/test_ui.py` defines a redundant reference class `SignLanguageHUDReference` and imports `RealtimeGestureRecognizer` rather than directly importing and testing `SignLanguageHUD` from `src/ui_overlay.py`.
- **Recommended Mitigation**: Refactor `tests/test_ui.py` to test `from ui_overlay import SignLanguageHUD, HUDState` directly.

---

## 2. Adversarial Stress Test Results Matrix

| Category | Test Scenario | Input / State | Expected | Actual | Status |
|---|---|---|---|---|---|
| **Resolution** | Tiny ($32\times32$) | $32\times32\times3$ BGR | No crash, safe clamping | No crash, full frame modified | **PASS** |
| **Resolution** | Tiny ($100\times100$) | $100\times100\times3$ BGR | No crash, safe clamping | No crash, rendered cleanly | **PASS** |
| **Resolution** | Standard ($640\times480$) | $480\times640\times3$ BGR | Clean rendering | Rendered correctly | **PASS** |
| **Resolution** | HD ($1280\times720$) | $720\times1280\times3$ BGR | Clean rendering | Rendered correctly | **PASS** |
| **Resolution** | Full HD ($1920\times1080$) | $1080\times1920\times3$ BGR | Clean rendering | Rendered correctly | **PASS** |
| **Resolution** | 4K UHD ($3840\times2160$) | $2160\times3840\times3$ BGR | Clean rendering | Rendered correctly | **PASS** |
| **Resolution** | Ultrawide ($2560\times400$) | $400\times2560\times3$ BGR | Safe clamping | Rendered correctly | **PASS** |
| **Resolution** | Tall ($400\times2560$) | $2560\times400\times3$ BGR | Safe clamping | Rendered correctly | **PASS** |
| **Coordinates** | Negative Bounding Box | `bbox=(-100, -50, 200, 300)` | Clamped safely | Clamped & rendered | **PASS** |
| **Coordinates** | Exceeding Frame Box | `bbox=(500, 400, 2000, 1500)` on $720\text{p}$ | Clamped safely | Clamped & rendered | **PASS** |
| **Coordinates** | Completely Out of Frame | `bbox=(-500, -500, -100, -100)` | No crash, clipped | Clipped safely | **PASS** |
| **Coordinates** | Inverted Bounding Box | `bbox=(400, 400, 200, 200)` | No crash | Handled gracefully | **PASS** |
| **Coordinates** | Zero-Area Bounding Box | `bbox=(100, 100, 100, 100)` | No crash | Handled gracefully | **PASS** |
| **Coordinates** | Out-of-Bounds Landmarks | Normalized coords $<0.0, >1.0$ | No crash | Clamped & rendered | **PASS** |
| **Coordinates** | Partial Landmarks (<21) | 5 landmarks provided | No crash, skip skeleton | Skipped safely | **PASS** |
| **Coordinates** | Empty Landmarks (`[]`) | `landmarks=[]`, no bbox | No crash, skip hand | `ValueError: min() arg is an empty sequence` | **FAIL** ❌ |
| **State** | Empty Class List | `classes=[]` | Skip legend | Skipped cleanly | **PASS** |
| **State** | None Class List | `classes=None` | Skip legend | Skipped cleanly | **PASS** |
| **State** | 100+ Class Names | `classes=[...]` (100 items) | Render without crash | Rendered, clipped to frame | **PASS** |
| **State** | 150-char Gesture Name | `gesture='A'*150` | Render without crash | Rendered across panel | **PASS** |
| **State** | 60-item Sequence | `sequence=[...]` (60 items) | Render without crash | Rendered string | **PASS** |
| **State** | Extreme FPS | `fps=0.0, 9999.9, -10.0` | Formats as string | Rendered `FPS: 0`, `FPS: 10000` | **PASS** |
| **State** | Inf / NaN FPS | `fps=float('inf'), float('nan')` | Formats safely | Rendered `FPS: inf`, `FPS: nan` | **PASS** |
| **State** | Negative Confidence | `confidence=-1.5` | Clamped to 0% | Rendered 0% | **PASS** |
| **State** | Over-range Confidence | `confidence=2.5` | Clamped to 100% | Rendered 100% | **PASS** |
| **State** | NaN Confidence | `confidence=float('nan')` | Clamped to 0% | Rendered 0% | **PASS** |
| **State** | Inf Confidence | `confidence=float('inf')` | Clamped to 100% | Rendered 100% | **PASS** |
| **State** | Handedness Score NaN | `score=float('nan')` | Safe fallback | `ValueError: cannot convert float NaN to integer` | **FAIL** ❌ |
| **State** | Handedness Score Inf | `score=float('inf')` | Safe fallback | `OverflowError: cannot convert float infinity to integer` | **FAIL** ❌ |
| **State** | None Handedness Score | `score=None` | Render label only | Rendered label only | **PASS** |
| **State** | Sequence Progress NaN/Inf | `sequence_progress=nan/inf` | Safe clamp | Rendered safely | **PASS** |
| **State** | Numeric Class Names | `classes=[0, 1, 2]` | Render labels | `AttributeError: 'int' object has no attribute 'capitalize'` | **FAIL** ❌ |
| **State** | `confidence=None` | `confidence=None` | Safe default 0.0 | `TypeError: float() argument must be a string or real number` | **FAIL** ❌ |
| **State** | `fps=None` | `fps=None` | Safe default 0.0 | `TypeError: unsupported format string` | **FAIL** ❌ |

---

## 3. Performance & Throughput Benchmark

### 3.1 Standalone `blend_roi` vs Full-Frame Compositing

| Resolution | Frame Size (Bytes) | Detection Panel ROI Size | Full-Frame Latency | ROI Latency | Speedup |
|---|---|---|---|---|---|
| **640x480** | 921.6 KB | $440 \times 60$ ($79.2\text{ KB}$) | **1.0188 ms** | **0.4832 ms** | **2.1x** |
| **1280x720 (720p)** | 2.76 MB | $1080 \times 60$ ($194.4\text{ KB}$) | **5.0572 ms** | **1.2026 ms** | **4.2x** |
| **1920x1080 (1080p)** | 6.22 MB | $1720 \times 60$ ($309.6\text{ KB}$) | **10.2356 ms** | **1.2570 ms** | **8.1x** |
| **3840x2160 (4K)** | 24.88 MB | $3640 \times 60$ ($655.2\text{ KB}$) | **36.6812 ms** | **2.8408 ms** | **12.9x** |

### 3.2 Full `SignLanguageHUD.render()` Latency (2 Hands + 6 Widgets)

| Resolution | In-Place Render Latency | Copy+Render Latency | Effective Render FPS | Max Camera Rate Supported |
|---|---|---|---|---|
| **640x480** | **5.67 ms** | **6.12 ms** | **163.5 FPS** | 60 FPS (172% headroom) |
| **1280x720 (720p)** | **7.62 ms** | **9.86 ms** | **101.5 FPS** | 60 FPS (69% headroom) |
| **1920x1080 (1080p)** | **8.93 ms** | **13.10 ms** | **76.4 FPS** | 60 FPS (27% headroom) |
| **3840x2160 (4K)** | **14.23 ms** | **31.43 ms** | **31.8 FPS** | 30 FPS (6% headroom) |

*Notes on throughput SLA*: Sub-array ROI blending for individual widgets takes **0.04 ms to 0.48 ms**, meeting sub-millisecond goals. The complete multi-widget HUD overlay (with anti-aliased geometry, multiple panels, text rendering, and 2 full hand skeletons) completes in **5.67 ms to 7.62 ms**, sustaining **100+ FPS** real-time video performance.

---

## 4. Required Action Items for Milestone M3 Sign-Off

1. **Fix Empty Landmark Guard**: In `SignLanguageHUD.draw_hands()` (`src/ui_overlay.py:237-254`), verify `lm_list` is non-empty before computing bounding box extents (`min(xs)` / `max(xs)`).
2. **Harden Handedness Badge against NaN/Inf**: In `SignLanguageHUD._draw_handedness_badge()` (`src/ui_overlay.py:147`), verify `score` is finite (`math.isfinite(score)`) before casting to `int`.
3. **Coerce Non-String Class Names and Gestures**: Use `str(name).capitalize()` in `draw_gesture_legend()` and `str(state.gesture).upper()` in `draw_detection_panel()`.
4. **Safely Default NoneType Confidence and FPS**: Provide fallback defaults (`0.0`) when `confidence` or `fps` is `None`.
5. **Update Test Suite**: Refactor `tests/test_ui.py` to directly import and test `SignLanguageHUD` from `src/ui_overlay.py` rather than maintaining a duplicate mock reference class.
