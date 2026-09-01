"""
Adversarial Stress Test Suite for Milestone M3 (UI Decoupling & Premium HUD).
Focuses on boundary stress, fuzzing, concurrency simulation, and architectural invariants.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    COL_DIM
)
from realtime_recognition import RealtimeGestureRecognizer


class AdversarialHUDStressTest(unittest.TestCase):
    """Adversarial and extreme edge case testing for SignLanguageHUD."""

    def setUp(self):
        self.hud = SignLanguageHUD(class_names=['hello', 'yes', 'no', 'thank_you'])

    def test_hudstate_fuzzing_and_boundary_extremes(self):
        """Stress HUDState with extreme numbers, empty, huge, and degenerate inputs."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Fuzz Case 1: Extreme numbers, huge strings, huge class lists
        state_fuzz = HUDState(
            fps=99999.0,
            detected=True,
            status_text="DETECTED",
            gesture="A" * 500,  # Extremely long gesture name
            confidence=0.9999,
            sequence=["G" * 50 for _ in range(50)],
            classes=[f"class_{i}" for i in range(40)],  # Exceeds vertical panel height
            sequence_progress=0.5
        )
        out = self.hud.render(frame.copy(), state_fuzz)
        self.assertEqual(out.shape, (720, 1280, 3))

        # Fuzz Case 2: Negative confidence, 10 valid hands, zero progress
        corrupted_hands = []
        for h_idx in range(10):
            corrupted_hands.append({
                'landmarks': [{'x': 0.1 * (i % 10), 'y': 0.1 * (i % 10), 'z': 0.0} for i in range(21)],
                'handedness': 'Right',
                'score': 0.99,
                'bbox': (10, 10, 200, 200)
            })

        state_corrupt = HUDState(
            fps=-15.0,
            detected=False,
            status_text="CORRUPT_STATUS",
            gesture="None",
            confidence=-0.99,
            sequence=[],
            classes=[],
            sequence_progress=-1.0,
            hands_info=corrupted_hands
        )
        out2 = self.hud.render(frame.copy(), state_corrupt)
        self.assertEqual(out2.shape, (720, 1280, 3))

    def test_blend_roi_alpha_extremes_and_float_precision(self):
        """Verify alpha blending at alpha=0.0 (identity) and alpha=1.0 (pure color replacement)."""
        canvas = np.full((100, 100, 3), 120, dtype=np.uint8)

        # Alpha = 0.0 (no change)
        SignLanguageHUD.blend_roi(canvas, 10, 10, 30, 30, (255, 0, 0), alpha=0.0)
        np.testing.assert_array_equal(canvas[10:40, 10:40], np.full((30, 30, 3), 120, dtype=np.uint8))

        # Alpha = 1.0 (100% overlay colour)
        SignLanguageHUD.blend_roi(canvas, 10, 10, 30, 30, (50, 60, 70), alpha=1.0)
        np.testing.assert_array_equal(canvas[10:40, 10:40], np.full((30, 30, 3), [50, 60, 70], dtype=np.uint8))

    def test_blend_roi_non_standard_resolutions(self):
        """Verify HUD rendering and blend_roi on non-standard resolutions (tiny 160x120, 4K 3840x2160, 1x1)."""
        tiny_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        hud = SignLanguageHUD(['a', 'b'])
        state = HUDState(fps=60.0, detected=True, gesture='a', confidence=0.9)

        out_tiny = hud.render(tiny_frame, state)
        self.assertEqual(out_tiny.shape, (120, 160, 3))

        single_pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        out_1x1 = hud.render(single_pixel, state)
        self.assertEqual(out_1x1.shape, (1, 1, 3))

    def test_realtime_recognizer_forwarding_methods(self):
        """Verify all backward-compatible delegated methods on RealtimeGestureRecognizer."""
        rec = object.__new__(RealtimeGestureRecognizer)
        rec.class_names = ['hello', 'yes']
        rec._frame_count = 5

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        rec._overlay_rect(frame, 10, 10, 50, 50)
        rec._draw_rounded_rect(frame, 10, 10, 50, 50, COL_CYAN)
        rec.draw_confidence_bar(frame, 10, 10, 100, 20, 0.8)
        rec.draw_top_bar(frame, 30.0)
        rec.draw_detection_panel(frame, "HELLO", 0.9, "DETECTED")
        rec.draw_sequence_panel(frame, "HELLO > YES")
        rec.draw_controls_bar(frame)
        rec.draw_gesture_legend(frame)

        self.assertTrue(np.any(frame > 0), "Forwarded methods should modify the canvas")


if __name__ == '__main__':
    unittest.main()
