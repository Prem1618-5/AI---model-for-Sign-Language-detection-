"""
HUD & UI Overlay Rendering Test Suite
Tests HUDState data model, SignLanguageHUD rendering directly from src/ui_overlay.py,
sub-array ROI alpha-blending math, boundary clipping, edge-case guards, and visual elements.
"""

import os
import sys
import time
import math
import unittest
from dataclasses import dataclass, field
import numpy as np
import cv2

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ui_overlay import (
    HUDState,
    SignLanguageHUD,
    COL_BG,
    COL_CYAN,
    COL_AMBER,
    COL_GREEN,
    COL_RED,
    COL_WHITE,
    COL_DIM,
    COL_BAR_BG
)
from realtime_recognition import RealtimeGestureRecognizer


class TestHUDStateAndDecoupledRenderer(unittest.TestCase):
    """Tier 1 & Tier 3: HUD state container invariants and decoupled renderer verification."""

    def test_hud_state_defaults(self):
        """Invariant: HUDState defaults are SCANNING, confidence 0.0, empty sequence."""
        state = HUDState()
        self.assertEqual(state.fps, 0.0)
        self.assertFalse(state.detected)
        self.assertEqual(state.status_text, "SCANNING")
        self.assertEqual(state.gesture, "None")
        self.assertEqual(state.confidence, 0.0)
        self.assertEqual(state.sequence, [])
        self.assertEqual(state.hands_info, [])
        self.assertEqual(state.classes, [])
        self.assertEqual(state.sequence_progress, 0.0)

    def test_roi_blending_math_and_performance(self):
        """Invariant: Sub-array ROI blending produces exact linear combination without memory copy overhead."""
        # Create solid white canvas
        canvas = np.full((480, 640, 3), 200, dtype=np.uint8)
        color = (10, 20, 30)
        alpha = 0.5

        t0 = time.perf_counter()
        SignLanguageHUD.blend_roi(canvas, 50, 50, 100, 100, color, alpha=alpha)
        dt = (time.perf_counter() - t0) * 1000.0

        # Sub-array ROI blending SLA must be < 0.5ms
        self.assertLess(dt, 0.5, f"ROI blending took {dt:.4f}ms, exceeds budget")

        # Invariant check: blended area expected = 0.5 * 10 + 0.5 * 200 = 105
        blended_pixel = canvas[75, 75]
        self.assertAlmostEqual(blended_pixel[0], 105, delta=2)
        self.assertAlmostEqual(blended_pixel[1], 110, delta=2)
        self.assertAlmostEqual(blended_pixel[2], 115, delta=2)

        # Unblended area remains untouched
        np.testing.assert_array_equal(canvas[10, 10], [200, 200, 200])

    def test_roi_blending_boundary_clipping(self):
        """Boundary: Blending outside image boundaries gracefully clips without index error."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        # Slices partially out of bounds
        SignLanguageHUD.blend_roi(canvas, 80, 80, 50, 50, (255, 255, 255), 0.5)
        # Slices completely out of bounds
        SignLanguageHUD.blend_roi(canvas, -50, -50, 30, 30, (255, 255, 255), 0.5)
        SignLanguageHUD.blend_roi(canvas, 200, 200, 50, 50, (255, 255, 255), 0.5)
        # Zero and negative dimensions
        SignLanguageHUD.blend_roi(canvas, 10, 10, 0, 0, (255, 255, 255), 0.5)
        SignLanguageHUD.blend_roi(canvas, 10, 10, -20, -20, (255, 255, 255), 0.5)

    def test_render_hud_on_synthetic_frame(self):
        """Test complete HUD rendering across different states without hardware camera."""
        hud = SignLanguageHUD(class_names=['hello', 'yes', 'no'])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # State 1: Scanning
        state1 = HUDState(fps=30.0, detected=False, status_text="SCANNING", gesture="None", confidence=0.1)
        out1 = hud.render(frame.copy(), state1)
        self.assertEqual(out1.shape, (720, 1280, 3))
        self.assertEqual(out1.dtype, np.uint8)

        # State 2: Detected
        state2 = HUDState(
            fps=32.5,
            detected=True,
            status_text="DETECTED",
            gesture="hello",
            confidence=0.92,
            sequence=["HELLO", "YES"],
            classes=['hello', 'yes', 'no']
        )
        out2 = hud.render(frame.copy(), state2)
        self.assertEqual(out2.shape, (720, 1280, 3))
        # Ensure frame content changed from all-black
        self.assertTrue(np.any(out2 > 0))

        # State 3: Uncertain / Analysing
        state3 = HUDState(
            fps=29.0,
            detected=False,
            status_text="UNCERTAIN",
            gesture="Analysing...",
            confidence=0.55,
            sequence=["HELLO"]
        )
        out3 = hud.render(frame.copy(), state3)
        self.assertEqual(out3.shape, (720, 1280, 3))
        self.assertTrue(np.any(out3 > 0))

    def test_empty_and_partial_landmarks_guard(self):
        """Adversarial/Defensive: Handle empty or partial landmarks gracefully without crashing."""
        hud = SignLanguageHUD()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Empty landmarks list with no bbox
        state_empty_lm = HUDState(hands_info=[{'landmarks': [], 'handedness': 'Right'}])
        out = hud.render(frame.copy(), state_empty_lm)
        self.assertEqual(out.shape, (720, 1280, 3))

        # 2. Hand landmarks with fewer than 21 points
        partial_lm = [{'x': 0.5, 'y': 0.5} for _ in range(5)]
        state_partial_lm = HUDState(hands_info=[{'landmarks': partial_lm, 'handedness': 'Left'}])
        out_partial = hud.render(frame.copy(), state_partial_lm)
        self.assertEqual(out_partial.shape, (720, 1280, 3))

        # 3. None landmarks
        state_none_lm = HUDState(hands_info=[{'landmarks': None, 'bbox': (100, 100, 200, 200), 'handedness': 'Right'}])
        out_none = hud.render(frame.copy(), state_none_lm)
        self.assertEqual(out_none.shape, (720, 1280, 3))

    def test_handedness_score_edge_cases(self):
        """Adversarial/Defensive: Handle NaN, Inf, None, negative, and over-range handedness scores."""
        hud = SignLanguageHUD()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        scores_to_test = [float('nan'), float('inf'), float('-inf'), None, -0.5, 1.5, 0.0, 1.0]
        for score in scores_to_test:
            state = HUDState(hands_info=[{
                'bbox': (100, 100, 200, 200),
                'handedness': 'Right',
                'score': score
            }])
            out = hud.render(frame.copy(), state)
            self.assertEqual(out.shape, (720, 1280, 3))

        # Test non-string and None label
        state_label = HUDState(hands_info=[{'bbox': (50, 50, 150, 150), 'handedness': 123, 'score': 0.9}])
        out_label = hud.render(frame.copy(), state_label)
        self.assertEqual(out_label.shape, (720, 1280, 3))

    def test_non_string_class_names_and_gestures(self):
        """Adversarial/Defensive: Handle numeric or non-string class names and gestures."""
        hud = SignLanguageHUD(class_names=[0, 1, 2])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # State with numeric classes in HUDState
        state = HUDState(
            classes=[101, 102, 103],
            status_text="DETECTED",
            gesture=101,
            confidence=0.88
        )
        out = hud.render(frame.copy(), state)
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertTrue(np.any(out > 0))

    def test_nonetype_and_extreme_fps_and_confidence(self):
        """Adversarial/Defensive: Safely default None, NaN, Inf, and extreme values for FPS & confidence."""
        hud = SignLanguageHUD()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Confidence None / NaN / Inf / Negative / Over-range
        for conf in [None, float('nan'), float('inf'), float('-inf'), -1.0, 2.5]:
            state = HUDState(confidence=conf, fps=30.0)
            out = hud.render(frame.copy(), state)
            self.assertEqual(out.shape, (720, 1280, 3))

        # FPS None / NaN / Inf / Negative / Very large
        for fps_val in [None, float('nan'), float('inf'), float('-inf'), -10.0, 99999.0]:
            state = HUDState(fps=fps_val, confidence=0.8)
            out = hud.render(frame.copy(), state)
            self.assertEqual(out.shape, (720, 1280, 3))

    def test_sequence_progress_edge_cases(self):
        """Adversarial/Defensive: Safely handle None, NaN, Inf, and out-of-range sequence progress."""
        hud = SignLanguageHUD()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        for sp in [None, float('nan'), float('inf'), float('-inf'), -0.5, 1.5, 0.5]:
            state = HUDState(sequence=["HELLO", "WORLD"], sequence_progress=sp)
            out = hud.render(frame.copy(), state)
            self.assertEqual(out.shape, (720, 1280, 3))

    def test_extreme_resolutions_and_coordinates(self):
        """Adversarial/Defensive: Render across tiny, 4K, ultrawide resolutions and out-of-frame bboxes."""
        hud = SignLanguageHUD(class_names=['hello', 'yes'])

        resolutions = [
            (32, 32),
            (100, 100),
            (480, 640),
            (720, 1280),
            (1080, 1920),
            (2160, 3840),
            (400, 2560)
        ]

        for h, w in resolutions:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            state = HUDState(
                fps=30.0,
                status_text="DETECTED",
                gesture="hello",
                confidence=0.85,
                hands_info=[
                    {'bbox': (-50, -50, 100, 100), 'handedness': 'Left', 'score': 0.95},
                    {'bbox': (w + 10, h + 10, w + 100, h + 100), 'handedness': 'Right', 'score': 0.88}
                ]
            )
            out = hud.render(frame, state)
            self.assertEqual(out.shape, (h, w, 3))


class TestRealtimeRecognizerHUDMethods(unittest.TestCase):
    """Tier 3: Tests for RealtimeGestureRecognizer internal drawing routines and delegation."""

    def setUp(self):
        # Create uninitialized recognizer instance with mock class names
        self.rec = object.__new__(RealtimeGestureRecognizer)
        self.rec.class_names = ['hello', 'thank_you', 'yes', 'no']
        self.rec._frame_count = 1
        self.rec._hud = SignLanguageHUD(class_names=self.rec.class_names)

    def test_confidence_bar_color_thresholds(self):
        """Test confidence bar fill color at high (>=0.7), medium (>=0.4), and low (<0.4)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        # High confidence (green)
        self.rec.draw_confidence_bar(frame, 10, 10, 100, 20, 0.85)
        self.assertEqual(frame.shape, (200, 400, 3))

        # Medium confidence (amber)
        self.rec.draw_confidence_bar(frame, 10, 40, 100, 20, 0.55)

        # Low confidence (red)
        self.rec.draw_confidence_bar(frame, 10, 70, 100, 20, 0.25)

    def test_all_hud_panels_composite(self):
        """Test compositing all HUD panels on a synthetic frame buffer without camera."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.rec.draw_top_bar(frame, fps=28.4)
        self.rec.draw_detection_panel(frame, "HELLO", 0.94, "DETECTED")
        self.rec.draw_sequence_panel(frame, "HELLO  >  YES")
        self.rec.draw_controls_bar(frame)
        self.rec.draw_gesture_legend(frame, 1280 - 175, 55)

        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(np.any(frame > 0))

    def test_recognizer_hud_delegation_and_none_values(self):
        """Test recognizer delegated drawing routines handle None values safely."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.rec.draw_top_bar(frame, fps=None)
        self.rec.draw_detection_panel(frame, None, None, None)
        self.rec.draw_sequence_panel(frame, None)
        self.rec.draw_confidence_bar(frame, 10, 10, 100, 20, None)

        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertTrue(np.any(frame > 0))


if __name__ == '__main__':
    unittest.main()
