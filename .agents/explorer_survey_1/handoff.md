# Handoff Report: Phase 0 Codebase Survey

**Agent**: `teamwork_preview_explorer` (Instance 1)  
**Date**: 2026-09-02  
**Working Directory**: `d:\Development Project\Sign Language\.agents\explorer_survey_1`  
**Full Report Path**: `d:\Development Project\Sign Language\.agents\explorer_survey_1\report.md`

---

## 1. Observation

1. **CLI Entry Point & Inconsistent Default Paths**:
   - In `src/main.py`: lines 30, 37, 39, 50, 52 set default path arguments to `'../data/raw'`, `'../data/processed'`, `'../models'`.
   - In `src/data_collection.py`: line 32 sets `output_dir='../data/raw'`; lines 46-47 call `os.makedirs(output_dir, exist_ok=True)` without checking if `data/raw` exists at project root.
   - In `src/realtime_recognition.py`: line 588 sets `models_dir = '../models'`.
   - Running `python src/main.py --help` from project root succeeded with exit code 0.

2. **Data Pipeline Schema Inconsistency & Silent Data Loss**:
   - `data/raw/` contains 7 JSON files: 3 generated with single-hand 2D format (`landmarks = [ [21 dicts], ... ]`) at `10:19:40`, and 4 collected with multi-hand 3D format (`landmarks = [ [ [21 dicts] ], ... ]`) at `10:30:25`–`10:31:49`.
   - In `src/data_preprocessing.py`: line 85 sets `is_two_handed = True` if any file has `'two_hands': True`.
   - In `src/data_preprocessing.py`: lines 239–288 check `if len(sample) == 1:` and `elif len(sample) == 2:`. For single-hand files, `len(sample) == 21`, which matches neither branch and results in all 150 single-hand samples being silently discarded.
   - Executing `.\venv\Scripts\python.exe src/main.py preprocess --input data/raw --output data/processed --augment` processed exactly 1,200 samples (4 files $\times$ 50 samples $\times$ 6 augmented = 1,200), confirming the other 3 files (150 samples) were omitted.

3. **Temporal Modeling State**:
   - In `src/model_training.py`: lines 122–127 define `lstm_input_shape = (1, input_shape[0])` and `layers.Reshape(lstm_input_shape, input_shape=input_shape)`. The LSTM receives sequence length 1.
   - In `src/realtime_recognition.py`: lines 88, 168–192 define `history_buffer = deque(maxlen=10)` and a 60% frequency majority vote. No probability smoothing or velocity/transition tracking is implemented.

4. **UI & Recognition Coupling**:
   - In `src/realtime_recognition.py`: 601 lines contain MediaPipe graph invocation, feature concatenation, Keras prediction, and 8 OpenCV overlay functions (`_overlay_rect`, `_draw_rounded_rect`, `draw_confidence_bar`, `draw_hand_skeleton`, `draw_gesture_legend`, `draw_top_bar`, `draw_detection_panel`, `draw_sequence_panel`, `draw_controls_bar`).

5. **Bloated & Unused Dependencies (Ponytail Violations)**:
   - In `src/data_preprocessing.py`: line 12 has `import pandas as pd`. Searching for `pd` in `src/` returned zero matches. `requirements.txt` line 2 lists `pandas==2.0.3`.
   - In `src/model_training.py`: line 16 has `import seaborn as sns`. Used exclusively on line 357 for `sns.heatmap`. `requirements.txt` line 4 lists `seaborn==0.12.2`.
   - In `src/data_collection.py`: line 15 has `from tqdm import tqdm`. `tqdm` is never called in `data_collection.py`.
   - In `MODULES_REFERENCE.md`: line 43 documents `extract_landmarks_from_file(filepath)` which does not exist in `src/data_collection.py`.

6. **Camera Diagnostic Tool**:
   - In `src/camera_test.py`: 42 lines implement `test_camera()`, opening `cv2.VideoCapture(0)` and streaming for 10 seconds or until `'q'`.

---

## 2. Logic Chain

1. From **Observation 1**, CLI defaults assume execution from `src/`, while user documentation (`README.md`) specifies execution from the workspace root. Without standardized path resolution (`os.path.abspath` or project-root defaults), commands executed from project root either create folders outside the project or fail.
2. From **Observation 2**, the branching logic in `data_preprocessing.py` assumes all samples conform to multi-hand list nesting when `is_two_handed` is true. Because legacy / synthetic single-hand files store samples as a flat list of 21 landmark objects, `len(sample) == 21`, bypassing both `len(sample) == 1` and `len(sample) == 2` branches. This directly leads to silent data loss during training data preparation.
3. From **Observation 3**, an LSTM with input shape `(1, 126)` has no temporal dimension across time steps ($T=1$). It functions as a feedforward network with recurrent computational overhead. The live inference relies solely on discrete modal filtering over a 10-frame buffer, resulting in boundary flicker between gesture transitions.
4. From **Observation 4**, combining drawing logic, device capture, and ML inference in `realtime_recognition.py` violates modularity (Requirement R2) and makes HUD visual improvements (Requirement R1) difficult to maintain or test in isolation.
5. From **Observation 5**, unused packages (`pandas`), single-use plot libraries (`seaborn`), and dead imports violate the Ponytail guideline of minimal dependencies, deletion over addition, and no unneeded abstractions.

---

## 3. Caveats

1. **Hardware Camera Validation**: A physical webcam was not actuated during this survey turn (headless environment). However, `src/camera_test.py` and `src/realtime_recognition.py` OpenCV loops were statically verified and traced against OpenCV 4.8.0 APIs.
2. **Data Scope**: The existing dataset in `data/raw` has 4 gesture classes (`hello`, `no`, `thanks`, `yes`). Dynamic signs requiring multi-frame movement are not yet part of the default raw dataset.

---

## 4. Conclusion

The sign language ML codebase is functional across its fundamental stages but requires targeted refactoring to meet the architectural, UI, temporal, and Ponytail requirements:
- **Modularity & UI (R1 & R2)**: Separate `realtime_recognition.py` into a recognition engine and a modular `HUDOverlay` drawing class with responsive positioning.
- **Data Pipeline (R2)**: Fix sample dimension parsing in `data_preprocessing.py` to transparently handle both 1-hand (`len==21`) and multi-hand (`len in (1,2)`) raw sample structures without dropping samples.
- **Temporal Stability (R3)**: Implement Exponential Moving Average (EMA) filtering on prediction class probabilities with confidence hysteresis to eliminate gesture flicker.
- **Code Cleanliness (R4)**: Remove `pandas` and `seaborn` dependencies; clean up unused imports and standardize CLI default paths to project root.

---

## 5. Verification Method

To independently verify the survey observations:
1. **Verify CLI Help**:
   ```powershell
   .\venv\Scripts\python.exe src/main.py --help
   ```
2. **Verify Preprocessing & Sample Discarding**:
   ```powershell
   .\venv\Scripts\python.exe src/main.py preprocess --input data/raw --output data/processed --augment
   ```
   Inspect stdout output: `Found 7 gesture data files` $\rightarrow$ `Processed dataset: 1200 samples`. (Confirms 150 single-hand samples were skipped).
3. **Verify Unused Imports**:
   Search for `pandas` / `pd` and `seaborn` / `sns`:
   ```powershell
   git grep -n "import pandas"
   git grep -n "pd\." src/
   git grep -n "import seaborn"
   ```
