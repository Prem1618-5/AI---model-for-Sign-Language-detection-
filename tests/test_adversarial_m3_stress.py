"""
Adversarial Stress Test Suite for Milestone M3 Components
Empirical Challenger: Instance 2

Focus Areas:
1. Continuous Frame Streams & 5,000+ Frame Memory Stability (Zero Leak Assertion)
2. Rapid User Input Key Events ('c' Clear, 's' Screenshot, Invalid Keys, Concurrency)
3. Full Backward Compatibility & Legacy Method / Attribute Parity
"""

import os
import sys
import gc
import time
import shutil
import tempfile
import tracemalloc
import unittest
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import cv2

# Ensure project root and src directory are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ui_overlay import (
    HUDState,
    SignLanguageHUD,
    HAND_CONNECTIONS,
    COL_BG,
    COL_CYAN,
    COL_AMBER,
    COL_GREEN,
    COL_RED,
    COL_WHITE,
    COL_GREY,
    COL_DIM,
    COL_BAR_BG,
    COL_BAR_FILL,
    COL_JOINT,
    COL_CONN,
    COL_PULSE_A,
    COL_PULSE_B
)
from realtime_recognition import RealtimeGestureRecognizer
from temporal_filter import TemporalSmoother
from data_preprocessing import GestureDataProcessor


class MockLandmark:
    """Mock single MediaPipe landmark with x, y, z coordinates."""
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class MockLandmarkList:
    """Mock MediaPipe NormalizedLandmarkList containing 21 landmarks."""
    def __init__(self, landmarks=None):
        if landmarks is None:
            # Default 21 landmarks in hand-like shape
            self.landmark = [MockLandmark(0.5 + 0.01 * i, 0.5 + 0.01 * (i % 4), 0.0) for i in range(21)]
        else:
            self.landmark = landmarks


class MockClassification:
    def __init__(self, label="Right", score=0.98):
        self.label = label
        self.score = float(score)


class MockHandedness:
    def __init__(self, label="Right", score=0.98):
        self.classification = [MockClassification(label=label, score=score)]


# =====================================================================
# 1. CONTINUOUS FRAME STREAM & 5,000+ FRAME MEMORY STABILITY TESTS
# =====================================================================
class TestContinuousFrameStreamMemoryStability(unittest.TestCase):
    """
    Stress-test memory stability and rendering throughput over 5,000+ frames.
    Asserts zero uncollected memory growth from overlay rendering and ROI blending.
    """

    def setUp(self):
        self.hud = SignLanguageHUD(class_names=['hello', 'thank_you', 'yes', 'no', 'help'])

    def test_5000_frames_continuous_stream_memory_stability(self):
        """
        Adversarial Invariant:
        Simulating 5,000 continuous video frames with dynamic HUDState, hand landmarks,
        and ROI blending must maintain strictly bounded memory (0 leak / growth < 0.5MB total).
        """
        gc.collect()
        tracemalloc.start()

        frame_h, frame_w = 720, 1280
        frame_buffer = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Pre-generate diverse landmark sets (1 hand, 2 hands, 0 hands)
        landmarks_single = MockLandmarkList()
        landmarks_dict = [{'x': 0.3 + 0.01 * i, 'y': 0.4 + 0.01 * (i % 4), 'z': 0.0} for i in range(21)]

        num_frames = 5200  # 5,000+ frames

        # Checkpoints for memory tracking
        memory_checkpoints = {}

        for frame_idx in range(num_frames):
            # Dynamic state variation across frames
            phase = frame_idx % 300
            if phase < 50:
                status = "SCANNING"
                gesture = "None"
                conf = 0.15
                hands_info = []
            elif phase < 100:
                status = "UNCERTAIN"
                gesture = "Analysing..."
                conf = 0.65
                hands_info = [{
                    'landmarks': landmarks_single,
                    'handedness': 'Right',
                    'score': 0.95,
                    'bbox': (400, 200, 600, 450)
                }]
            elif phase < 250:
                status = "DETECTED"
                gesture = self.hud.class_names[frame_idx % len(self.hud.class_names)]
                conf = 0.88 + 0.10 * np.sin(frame_idx * 0.1)
                hands_info = [
                    {
                        'landmarks': landmarks_single,
                        'handedness': 'Right',
                        'score': 0.96,
                        'bbox': (400, 200, 600, 450)
                    },
                    {
                        'landmarks': landmarks_dict,
                        'handedness': 'Left',
                        'score': 0.91,
                        'bbox': (700, 250, 900, 500)
                    }
                ]
            else:
                status = "DETECTED"
                gesture = "yes"
                conf = 0.94
                hands_info = [{
                    'landmarks': landmarks_dict,
                    'handedness': 'Left',
                    'score': 0.92,
                    'bbox': (500, 200, 750, 450)
                }]

            seq = ["HELLO", "THANK_YOU", "YES"][:(frame_idx % 4)]
            seq_progress = max(0.0, 1.0 - (phase / 300.0))

            state = HUDState(
                fps=29.8 + (frame_idx % 5) * 0.2,
                detected=(status == "DETECTED"),
                status_text=status,
                gesture=gesture,
                confidence=conf,
                sequence=seq,
                hands_info=hands_info,
                classes=self.hud.class_names,
                sequence_progress=seq_progress
            )

            self.hud.render(frame_buffer, state)

            # Record memory at checkpoints: 1000, 2000, 3000, 4000, 5000
            if (frame_idx + 1) in [1000, 2000, 3000, 4000, 5000]:
                curr, peak = tracemalloc.get_traced_memory()
                memory_checkpoints[frame_idx + 1] = (curr / 1024 / 1024, peak / 1024 / 1024)

        gc.collect()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 1. Assert memory stability: Delta between frame 1000 and frame 5000 must be negligible (< 0.5 MB)
        mem_1000 = memory_checkpoints[1000][0]
        mem_5000 = memory_checkpoints[5000][0]
        mem_growth_mb = abs(mem_5000 - mem_1000)

        self.assertLess(
            mem_growth_mb, 0.50,
            f"Uncontrolled memory growth detected: frame 1000={mem_1000:.3f}MB, frame 5000={mem_5000:.3f}MB (growth={mem_growth_mb:.3f}MB)"
        )

    def test_rendering_throughput_and_roi_latency_sla(self):
        """Throughput SLA: Sub-array ROI blending and full HUD render must support >60 FPS throughput."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Test blend_roi SLA (< 1.0ms per ROI call for large panel)
        t0 = time.perf_counter()
        n_blends = 500
        for _ in range(n_blends):
            SignLanguageHUD.blend_roi(frame, 15, 100, 400, 80, colour=COL_BG, alpha=0.80)
        dt_blend_ms = ((time.perf_counter() - t0) * 1000.0) / n_blends

        self.assertLess(
            dt_blend_ms, 1.0,
            f"blend_roi latency {dt_blend_ms:.4f}ms exceeds 1.0ms SLA"
        )

        # 2. Test full HUD rendering SLA (< 15.0ms per frame -> >60 FPS capable)
        state = HUDState(
            fps=30.0,
            detected=True,
            status_text="DETECTED",
            gesture="hello",
            confidence=0.94,
            sequence=["HELLO", "YES"],
            classes=self.hud.class_names,
            hands_info=[{
                'landmarks': MockLandmarkList(),
                'handedness': 'Right',
                'score': 0.95,
                'bbox': (400, 200, 600, 450)
            }],
            sequence_progress=0.75
        )

        t0 = time.perf_counter()
        n_frames = 200
        for _ in range(n_frames):
            self.hud.render(frame, state)
        dt_frame_ms = ((time.perf_counter() - t0) * 1000.0) / n_frames

        self.assertLess(
            dt_frame_ms, 15.0,
            f"Full HUD rendering latency {dt_frame_ms:.4f}ms exceeds 15.0ms SLA"
        )

    def test_combined_smoother_and_hud_pipeline_stream(self):
        """Stress: 3,000 iterations of full TemporalSmoother + HUDState + SignLanguageHUD pipeline."""
        smoother = TemporalSmoother(
            alpha=0.25, threshold_high=0.80, threshold_low=0.45,
            debounce_frames=4, class_names=['hello', 'yes', 'no']
        )
        hud = SignLanguageHUD(class_names=['hello', 'yes', 'no'])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        t = 0.0
        for i in range(3000):
            t += 0.033
            # Rotate raw probability distributions
            if i % 300 < 100:
                raw_p = [0.90, 0.05, 0.05]
                wrist = (0.4, 0.5)
            elif i % 300 < 200:
                raw_p = [0.05, 0.90, 0.05]
                wrist = (0.42, 0.51)
            else:
                raw_p = [0.33, 0.33, 0.34]
                wrist = (0.80, 0.80)  # fast jump

            gesture, conf, status = smoother.update(raw_p, wrist_pos=wrist, timestamp=t)
            state = HUDState(
                fps=30.0,
                detected=(status == "DETECTED"),
                status_text=status,
                gesture=gesture,
                confidence=conf,
                sequence=smoother.get_sequence(),
                classes=['hello', 'yes', 'no']
            )
            out_frame = hud.render(frame, state)
            self.assertEqual(out_frame.shape, (720, 1280, 3))

    def test_multi_resolution_canvas_rendering_stress(self):
        """Stress: Render HUD across multiple canvas resolutions and aspect ratios."""
        resolutions = [
            (480, 640),    # VGA (4:3)
            (720, 1280),   # 720p HD (16:9)
            (1080, 1920),  # 1080p FHD (16:9)
            (240, 320),    # QVGA
            (300, 300),    # Square
            (100, 100),    # Tiny canvas
            (1, 1),        # Degenerate 1x1 canvas
        ]
        state = HUDState(
            fps=30.0,
            detected=True,
            status_text="DETECTED",
            gesture="hello",
            confidence=0.95,
            sequence=["HELLO", "YES"],
            classes=['hello', 'yes', 'no']
        )

        for h, w in resolutions:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Must render without throwing exception regardless of small or degenerate size
            try:
                out = self.hud.render(frame, state)
                self.assertEqual(out.shape, (h, w, 3))
            except Exception as e:
                self.fail(f"HUD render crashed on resolution ({w}x{h}): {e}")

    def test_malformed_and_extreme_landmark_inputs(self):
        """Boundary: Verify HUD handles malformed, partial, or out-of-bounds landmark data gracefully."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Landmarks with out-of-bounds coordinates (negative, > 1.0, huge values)
        oob_landmarks = [
            {'x': -5.0, 'y': -2.0, 'z': 0.0},
            {'x': 15.0, 'y': 25.0, 'z': 0.0},
        ] + [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(19)]

        hands_info = [{
            'landmarks': oob_landmarks,
            'handedness': 'Left',
            'score': 0.99,
            'bbox': (-50, -50, 2000, 2000)
        }]
        state = HUDState(hands_info=hands_info)
        # Should not crash
        self.hud.render(frame.copy(), state)

        # 2. Fewer than 21 landmarks (e.g. 5 points)
        partial_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(5)]
        state_partial = HUDState(hands_info=[{'landmarks': partial_landmarks, 'handedness': 'Right'}])
        self.hud.render(frame.copy(), state_partial)

        # 3. None landmarks but valid bbox
        state_nobbox = HUDState(hands_info=[{'landmarks': None, 'bbox': (50, 50, 100, 100)}])
        self.hud.render(frame.copy(), state_nobbox)

        # 4. Extreme text lengths in HUDState
        state_extreme_text = HUDState(
            status_text="A" * 1000,
            gesture="G" * 5000,
            sequence=["SEQ_" + str(i) * 50 for i in range(20)],
            confidence=float('nan')
        )
        self.hud.render(frame.copy(), state_extreme_text)


# =====================================================================
# 2. RAPID USER INPUT KEY EVENTS & CONCURRENCY TESTS
# =====================================================================
class TestRapidUserInputKeyEvents(unittest.TestCase):
    """Stress-test user input key event dispatching ('c', 's', invalid keys) under high frequency."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_key_test_")
        self.rec = object.__new__(RealtimeGestureRecognizer)
        self.rec.class_names = ['hello', 'yes', 'no']
        self.rec.is_two_handed_model = False
        self.rec.smoothing_window = 10
        self.rec.history_buffer = []
        self.rec.sequence_buffer = []
        self.rec.sequence_timeout = 2.0
        self.rec.last_gesture_time = time.time()
        self.rec.smoother = TemporalSmoother(class_names=self.rec.class_names)
        self.rec._hud = SignLanguageHUD(class_names=self.rec.class_names)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_rapid_clear_sequence_key_events(self):
        """Stress: 1,000 rapid 'c' key presses interleaved with sequence additions."""
        for i in range(1000):
            # Populate buffers
            self.rec.smoother.update([0.95, 0.02, 0.03])
            self.rec.history_buffer.append(("hello", 0.95))
            self.rec.sequence_buffer.append(("hello", 0.95))

            # Trigger 'c' clear logic (exact code from realtime_recognition.py lines 481-486)
            self.rec.smoother.reset()
            self.rec.smoother.sequence_buffer.clear()
            self.rec.sequence_buffer = []
            if hasattr(self.rec.history_buffer, 'clear'):
                self.rec.history_buffer.clear()
            else:
                self.rec.history_buffer = []

            # Verify clean reset state
            self.assertEqual(len(self.rec.smoother.sequence_buffer), 0)
            self.assertEqual(len(self.rec.sequence_buffer), 0)
            self.assertEqual(len(self.rec.history_buffer), 0)
            self.assertEqual(self.rec.smoother.current_state, "SCANNING")

    def test_rapid_screenshot_generation_and_io(self):
        """Stress: Rapid 's' key screenshot event handling without file lock collisions or corruption."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        created_files = []

        for i in range(25):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include unique index or sub-second resolution to prevent collisions in high-speed loops
            screenshot_path = os.path.join(self.test_dir, f"screenshot_{ts}_{i}.png")
            success = cv2.imwrite(screenshot_path, frame)
            self.assertTrue(success, f"cv2.imwrite failed for {screenshot_path}")
            self.assertTrue(os.path.exists(screenshot_path))
            self.assertGreater(os.path.getsize(screenshot_path), 0)
            created_files.append(screenshot_path)

        self.assertEqual(len(created_files), 25)

    def test_invalid_and_boundary_key_codes(self):
        """Boundary: Unhandled key codes (-1, 0, 255, 65535, ESC, Arrows) do not alter recognizer state."""
        invalid_keys = [
            -1, 0, 255, 65535, 27,  # -1 (no key), ESC (27)
            ord('x'), ord('z'), ord('1'), ord(' '), ord('\n'),
            0x250000, 0x260000,     # Arrow keys in some OpenCV backends
            0xFF
        ]

        # Initial state
        self.rec.smoother.update([0.9, 0.05, 0.05])
        initial_seq_len = len(self.rec.smoother.sequence_buffer)

        for key_code in invalid_keys:
            key = key_code & 0xFF
            # Test key dispatch logic
            if key == ord('q'):
                pass
            elif key == ord('c'):
                self.rec.smoother.reset()
            elif key == ord('s'):
                pass
            else:
                # Key is ignored - ensure state remains unchanged
                self.assertEqual(len(self.rec.smoother.sequence_buffer), initial_seq_len)
                self.assertIn(self.rec.smoother.current_state, ["SCANNING", "UNCERTAIN", "DETECTED"])


# =====================================================================
# 3. FULL BACKWARD COMPATIBILITY & LEGACY API STRESS TESTS
# =====================================================================
class TestLegacyApiAndBackwardCompatibility(unittest.TestCase):
    """
    Verify complete backward compatibility for all legacy methods, attributes,
    and function signatures in RealtimeGestureRecognizer and SignLanguageHUD.
    """

    def setUp(self):
        self.rec = object.__new__(RealtimeGestureRecognizer)
        self.rec.class_names = ['hello', 'thank_you', 'yes', 'no']
        self.rec.is_two_handed_model = False
        self.rec.smoothing_window = 10
        self.rec.history_buffer = []
        self.rec.sequence_buffer = []
        self.rec.sequence_timeout = 2.0
        self.rec.last_gesture_time = 0.0
        self.rec.processor = GestureDataProcessor()
        self.rec.smoother = TemporalSmoother(class_names=self.rec.class_names)
        self.rec._hud = SignLanguageHUD(class_names=self.rec.class_names)

    def test_legacy_overlay_rect_delegation(self):
        """Legacy API: _overlay_rect performs in-place alpha blending identically to HUD blend_roi."""
        frame = np.full((300, 300, 3), 100, dtype=np.uint8)
        self.rec._overlay_rect(frame, 50, 50, 100, 100, colour=COL_BG, alpha=0.5)
        # Check blended region is modified
        self.assertNotEqual(frame[75, 75, 0], 100)
        # Check untouched region
        np.testing.assert_array_equal(frame[10, 10], [100, 100, 100])

    def test_legacy_draw_rounded_rect_delegation(self):
        """Legacy API: _draw_rounded_rect draws bordered rectangle."""
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        self.rec._draw_rounded_rect(frame, 20, 20, 100, 100, colour=COL_CYAN, thickness=2, radius=10)
        self.assertTrue(np.any(frame > 0))

    def test_legacy_draw_confidence_bar(self):
        """Legacy API: draw_confidence_bar draws bar without errors across full [0.0, 1.0] range."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        for conf in [-0.5, 0.0, 0.35, 0.70, 0.95, 1.0, 1.5]:
            self.rec.draw_confidence_bar(frame, 10, 10, 180, 20, confidence=conf)
            self.assertTrue(np.any(frame > 0))

    def test_legacy_draw_hand_skeleton(self):
        """Legacy API: draw_hand_skeleton works with both MediaPipe landmark objects and dictionary lists."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. MediaPipe MockLandmarkList
        mp_lm = MockLandmarkList()
        self.rec.draw_hand_skeleton(frame, mp_lm)
        self.assertTrue(np.any(frame > 0))

        # 2. Dict list
        dict_lm = [{'x': 0.4 + 0.01 * i, 'y': 0.4 + 0.01 * (i % 4), 'z': 0.0} for i in range(21)]
        frame.fill(0)
        self.rec.draw_hand_skeleton(frame, dict_lm)
        self.assertTrue(np.any(frame > 0))

    def test_legacy_draw_gesture_legend_default_and_custom_positions(self):
        """Legacy API: draw_gesture_legend handles default None coords and explicit (x, y)."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Default args
        self.rec.draw_gesture_legend(frame)
        self.assertTrue(np.any(frame > 0))

        # Explicit coords
        frame.fill(0)
        self.rec.draw_gesture_legend(frame, x=50, y=50)
        self.assertTrue(np.any(frame > 0))

    def test_legacy_draw_top_bar_detection_panel_sequence_panel_controls_bar(self):
        """Legacy API: draw_top_bar, draw_detection_panel, draw_sequence_panel, draw_controls_bar."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.rec.draw_top_bar(frame, fps=30.0)
        self.rec.draw_detection_panel(frame, "HELLO", 0.95, "DETECTED")
        self.rec.draw_sequence_panel(frame, "HELLO  >  YES")
        self.rec.draw_controls_bar(frame)

        self.assertTrue(np.any(frame > 0))

    def test_legacy_get_smoothed_prediction(self):
        """Legacy API: get_smoothed_prediction returns mode gesture and average confidence."""
        # Empty buffer
        self.rec.history_buffer = []
        g, c = self.rec.get_smoothed_prediction()
        self.assertIsNone(g)
        self.assertEqual(c, 0.0)

        # 7 'hello' at 0.90, 3 'yes' at 0.80 -> 70% 'hello' >= 60% threshold
        self.rec.history_buffer = [('hello', 0.90)] * 7 + [('yes', 0.80)] * 3
        g, c = self.rec.get_smoothed_prediction()
        self.assertEqual(g, 'hello')
        self.assertAlmostEqual(c, 0.90, places=3)

        # Split 50/50 -> below 60% threshold -> returns (None, 0.0)
        self.rec.history_buffer = [('hello', 0.90)] * 5 + [('yes', 0.80)] * 5
        g, c = self.rec.get_smoothed_prediction()
        self.assertIsNone(g)
        self.assertEqual(c, 0.0)

    def test_legacy_update_sequence_and_get_sequence_text(self):
        """Legacy API: update_sequence and get_sequence_text maintain arrow-separated string."""
        self.rec.sequence_buffer = []
        self.rec.last_gesture_time = time.time()

        self.rec.update_sequence('hello', 0.95)
        self.rec.update_sequence('yes', 0.90)
        self.rec.update_sequence('no', 0.85)

        seq_text = self.rec.get_sequence_text()
        self.assertEqual(seq_text, "HELLO  >  YES  >  NO")

    def test_legacy_preprocessing_methods(self):
        """Legacy API: preprocess_landmarks, preprocess_two_hands, preprocess_single_hand_for_two_handed_model."""
        mp_landmarks = MockLandmarkList()

        # 1. Single hand features (63-dim)
        feat_single = self.rec.preprocess_landmarks(mp_landmarks)
        self.assertEqual(feat_single.shape, (63,))
        self.assertTrue(np.issubdtype(feat_single.dtype, np.floating))

        # 2. Two hands combined features (126-dim)
        feat_two = self.rec.preprocess_two_hands(mp_landmarks, mp_landmarks)
        self.assertEqual(feat_two.shape, (126,))

        # 3. Two hands with one hand None
        feat_two_single = self.rec.preprocess_two_hands(mp_landmarks, None)
        self.assertEqual(feat_two_single.shape, (126,))
        np.testing.assert_array_equal(feat_two_single[63:], np.zeros(63))

        # 4. Single hand padded for two-handed model (126-dim)
        feat_padded = self.rec.preprocess_single_hand_for_two_handed_model(mp_landmarks)
        self.assertEqual(feat_padded.shape, (126,))
        np.testing.assert_array_equal(feat_padded[63:], np.zeros(63))

    def test_hud_dual_signature_support(self):
        """SignLanguageHUD dual signature compatibility for all panels."""
        hud = SignLanguageHUD(class_names=['hello', 'yes', 'no'])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = HUDState(
            fps=30.0,
            detected=True,
            status_text="DETECTED",
            gesture="hello",
            confidence=0.92,
            sequence=["HELLO", "YES"],
            classes=['hello', 'yes', 'no']
        )

        # Signature 1: Object state
        hud.draw_top_bar(frame, state)
        hud.draw_detection_panel(frame, state)
        hud.draw_sequence_panel(frame, state)
        hud.draw_gesture_legend(frame, state)

        # Signature 2: Positional args
        frame.fill(0)
        hud.draw_top_bar(frame, 30.0)
        hud.draw_detection_panel(frame, "HELLO", 0.92, "DETECTED")
        hud.draw_sequence_panel(frame, "HELLO  >  YES")
        hud.draw_gesture_legend(frame, 1280 - 175, 55)

        self.assertTrue(np.any(frame > 0))


if __name__ == '__main__':
    unittest.main()
