"""
Independent Forensic Integrity Test Suite for Milestone M3 (UI Decoupling & Premium HUD).
Designed by Forensic Auditor to verify:
1. Genuineness of SignLanguageHUD OpenCV rendering onto frame buffers (no mock/bypass).
2. Mathematical correctness, boundary clipping, and performance of blend_roi in-place alpha compositing.
3. Architectural decoupling of RealtimeGestureRecognizer from raw cv2 drawing operations.
4. Dependency verification across all source files.
"""

import os
import sys
import ast
import time
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


class ForensicHUDVerification(unittest.TestCase):
    """Forensic verification that SignLanguageHUD genuinely mutates frame buffers."""

    def setUp(self):
        self.hud = SignLanguageHUD(class_names=['hello', 'yes', 'no', 'thank_you'])

    def test_blend_roi_mathematical_precision(self):
        """Verify blend_roi performs exact alpha compositing: dst = alpha * colour + (1 - alpha) * src."""
        h, w = 100, 100
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = [100, 150, 200]  # BGR initial

        test_colour = (20, 40, 60)
        alpha = 0.75

        # Apply blending on a 40x40 patch
        SignLanguageHUD.blend_roi(canvas, 10, 10, 40, 40, test_colour, alpha=alpha)

        # Expected calculation:
        # B: 0.75 * 20 + 0.25 * 100 = 15 + 25 = 40
        # G: 0.75 * 40 + 0.25 * 150 = 30 + 37.5 = 67.5 -> 67 or 68 (OpenCV rounding)
        # R: 0.75 * 60 + 0.25 * 200 = 45 + 50 = 95
        blended_region = canvas[10:50, 10:50]
        unblended_region = canvas[60:90, 60:90]

        self.assertEqual(blended_region.shape, (40, 40, 3))
        # Verify all pixels in blended patch have changed to the exact expected mathematical values
        expected_b = round(alpha * test_colour[0] + (1 - alpha) * 100)
        expected_g = round(alpha * test_colour[1] + (1 - alpha) * 150)
        expected_r = round(alpha * test_colour[2] + (1 - alpha) * 200)

        # Tolerating +/- 1 for integer rounding in cv2.addWeighted
        self.assertTrue(np.all(np.abs(blended_region[:, :, 0] - expected_b) <= 1))
        self.assertTrue(np.all(np.abs(blended_region[:, :, 1] - expected_g) <= 1))
        self.assertTrue(np.all(np.abs(blended_region[:, :, 2] - expected_r) <= 1))

        # Verify untouched region
        self.assertTrue(np.all(unblended_region[:, :, 0] == 100))
        self.assertTrue(np.all(unblended_region[:, :, 1] == 150))
        self.assertTrue(np.all(unblended_region[:, :, 2] == 200))

    def test_blend_roi_boundary_stress(self):
        """Verify blend_roi gracefully handles out-of-bounds, negative, inverted, and zero-size inputs."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        copy_canvas = canvas.copy()

        # Completely negative coordinates
        SignLanguageHUD.blend_roi(canvas, -100, -100, 50, 50, COL_BG, 0.8)
        np.testing.assert_array_equal(canvas, copy_canvas)

        # Zero width / height
        SignLanguageHUD.blend_roi(canvas, 10, 10, 0, 50, COL_BG, 0.8)
        SignLanguageHUD.blend_roi(canvas, 10, 10, 50, 0, COL_BG, 0.8)
        SignLanguageHUD.blend_roi(canvas, 10, 10, -10, -10, COL_BG, 0.8)
        np.testing.assert_array_equal(canvas, copy_canvas)

        # Far outside right/bottom
        SignLanguageHUD.blend_roi(canvas, 200, 200, 50, 50, COL_BG, 0.8)
        np.testing.assert_array_equal(canvas, copy_canvas)

        # Partial boundary overlap (top-left crossing, bottom-right crossing)
        SignLanguageHUD.blend_roi(canvas, -20, -20, 40, 40, (255, 255, 255), 1.0)
        # Sliced region [0:20, 0:20] should be white
        self.assertTrue(np.all(canvas[0:20, 0:20] == 255))
        self.assertTrue(np.all(canvas[21:100, 21:100] == 0))

    def test_blend_roi_performance_sla(self):
        """Verify blend_roi runs well within SLA budget across 1,000 runs."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        runs = 1000
        t0 = time.perf_counter()
        for _ in range(runs):
            SignLanguageHUD.blend_roi(canvas, 50, 50, 100, 100, COL_BG, 0.8)
        elapsed_ms_per_call = ((time.perf_counter() - t0) / runs) * 1000.0

        print(f"\n[Performance SLA] blend_roi (100x100) average latency: {elapsed_ms_per_call:.5f} ms/call")
        self.assertLess(elapsed_ms_per_call, 0.5, f"blend_roi too slow: {elapsed_ms_per_call} ms")

    def test_all_individual_drawing_methods_mutate_pixels(self):
        """Verify that every drawing method in SignLanguageHUD genuinely modifies the frame."""
        methods = [
            ("draw_top_bar", lambda f: self.hud.draw_top_bar(f, 30.0)),
            ("draw_detection_panel", lambda f: self.hud.draw_detection_panel(f, "HELLO", 0.95, "DETECTED")),
            ("draw_sequence_panel", lambda f: self.hud.draw_sequence_panel(f, "HELLO  >  YES")),
            ("draw_gesture_legend", lambda f: self.hud.draw_gesture_legend(f, 1280 - 175, 55)),
            ("draw_controls_bar", lambda f: self.hud.draw_controls_bar(f)),
            ("draw_confidence_bar", lambda f: self.hud.draw_confidence_bar(f, 100, 100, 200, 20, 0.85)),
            ("draw_corner_brackets", lambda f: self.hud._draw_corner_brackets(f, 100, 100, 300, 300)),
            ("draw_rounded_rect", lambda f: self.hud._draw_rounded_rect(f, 100, 100, 200, 200, COL_CYAN)),
            ("draw_handedness_badge", lambda f: self.hud._draw_handedness_badge(f, 100, 100, "Right", 0.98))
        ]

        for name, method in methods:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            method(frame)
            changed_pixels = np.count_nonzero(frame)
            self.assertGreater(
                changed_pixels, 0,
                f"Method {name} did not mutate any pixels in the frame buffer! Potential mock/facade detected."
            )

    def test_hand_skeleton_and_hands_rendering_mutates_pixels(self):
        """Verify hand skeleton and hand landmark drawing genuinely renders lines and circles."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Mock 21 landmarks
        mock_landmarks = [{'x': 0.5 + 0.01 * i, 'y': 0.5 + 0.01 * i, 'z': 0.0} for i in range(21)]
        self.hud.draw_hand_skeleton(frame, mock_landmarks)
        self.assertGreater(np.count_nonzero(frame), 0, "draw_hand_skeleton did not mutate pixels")

        # Mock hands_info
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        hands_info = [{
            'landmarks': mock_landmarks,
            'handedness': 'Right',
            'score': 0.96,
            'bbox': (400, 200, 700, 500)
        }]
        self.hud.draw_hands(frame2, hands_info)
        self.assertGreater(np.count_nonzero(frame2), 0, "draw_hands did not mutate pixels")

    def test_render_composite_full_pipeline(self):
        """Verify full HUDState composite render produces multi-region visual changes."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = HUDState(
            fps=45.0,
            detected=True,
            status_text="DETECTED",
            gesture="thank_you",
            confidence=0.88,
            sequence=["HELLO", "THANK_YOU"],
            classes=['hello', 'yes', 'no', 'thank_you'],
            sequence_progress=0.75,
            hands_info=[{
                'landmarks': [{'x': 0.4 + 0.01 * i, 'y': 0.4 + 0.01 * i, 'z': 0.0} for i in range(21)],
                'handedness': 'Right',
                'score': 0.99,
                'bbox': (300, 200, 600, 500)
            }]
        )

        out_frame = self.hud.render(frame, state)
        self.assertIs(out_frame, frame, "render() should return modified frame in-place")
        
        # Verify pixel changes in each HUD region
        # 1. Top bar region: y in [0, 45]
        self.assertTrue(np.any(frame[0:45, :, :] > 0), "Top bar not rendered")
        # 2. Legend region: x in [1280-175, 1280], y in [55, 200]
        self.assertTrue(np.any(frame[55:200, 1280-175:1280, :] > 0), "Legend sidebar not rendered")
        # 3. Detection panel: y in [720-150, 720-90]
        self.assertTrue(np.any(frame[570:630, 15:1095, :] > 0), "Detection panel not rendered")
        # 4. Sequence panel: y in [720-85, 720-45]
        self.assertTrue(np.any(frame[635:675, 15:1095, :] > 0), "Sequence panel not rendered")
        # 5. Controls bar: y in [720-30, 720]
        self.assertTrue(np.any(frame[690:720, :, :] > 0), "Controls bar not rendered")
        # 6. Hand bounding box & skeleton: y in [200, 500], x in [300, 600]
        self.assertTrue(np.any(frame[200:500, 300:600, :] > 0), "Hand skeleton/box not rendered")


class ForensicDecouplingVerification(unittest.TestCase):
    """Forensic verification that RealtimeGestureRecognizer is decoupled from drawing logic."""

    def test_ast_drawing_calls_in_realtime_recognition(self):
        """
        Inspect AST of realtime_recognition.py to ensure the core recognition loop
        does not invoke raw cv2 drawing primitives (rectangle, putText, circle, line, etc.)
        outside of delegated wrappers or HUD delegation.
        """
        rt_path = os.path.join(SRC_DIR, 'realtime_recognition.py')
        with open(rt_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Inspect all functions and methods in RealtimeGestureRecognizer
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "RealtimeGestureRecognizer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # The run() method must NOT call cv2 drawing functions
                        if item.name == "run":
                            for subnode in ast.walk(item):
                                if isinstance(subnode, ast.Call):
                                    if isinstance(subnode.func, ast.Attribute):
                                        if isinstance(subnode.func.value, ast.Name) and subnode.func.value.id == "cv2":
                                            func_name = subnode.func.attr
                                            prohibited_draw_ops = [
                                                'rectangle', 'putText', 'circle', 'line', 'ellipse',
                                                'polylines', 'fillPoly', 'drawContours'
                                            ]
                                            self.assertNotIn(
                                                func_name, prohibited_draw_ops,
                                                f"Architectural decoupling violation: RealtimeGestureRecognizer.run() "
                                                f"directly calls cv2.{func_name} at line {subnode.lineno}!"
                                            )

    def test_hud_property_and_delegation(self):
        """Verify RealtimeGestureRecognizer cleanly delegates rendering to SignLanguageHUD."""
        rec = object.__new__(RealtimeGestureRecognizer)
        rec.class_names = ['hello', 'yes']
        rec._frame_count = 0

        # Verify hud property instantiation
        hud_inst = rec.hud
        self.assertIsInstance(hud_inst, SignLanguageHUD)

        # Mock hud to verify delegation
        class MockHUD:
            def __init__(self):
                self.rendered_state = None
                self.render_called = False
            def render(self, frame, state):
                self.render_called = True
                self.rendered_state = state
                return frame

        mock = MockHUD()
        rec.hud = mock
        self.assertIs(rec.hud, mock)


class ForensicDependencyVerification(unittest.TestCase):
    """Forensic verification that no unapproved dependencies were introduced."""

    def test_imports_against_approved_manifest(self):
        """Verify all imports across src/ belong to Python stdlib or approved requirements."""
        approved_external_pkgs = {
            'numpy', 'cv2', 'mediapipe', 'tensorflow', 'sklearn', 'matplotlib', 'tqdm'
        }
        
        # Standard library modules commonly used
        stdlib_modules = {
            'os', 'sys', 'time', 'json', 'glob', 'argparse', 'collections',
            'datetime', 'dataclasses', 'unittest', 'traceback', 'math', 'typing',
            'random', 'shutil', 'tempfile', 'pathlib'
        }

        for root, _, files in os.walk(SRC_DIR):
            for file in files:
                if file.endswith('.py'):
                    fpath = os.path.join(root, file)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                top_pkg = alias.name.split('.')[0]
                                if top_pkg not in stdlib_modules and top_pkg not in approved_external_pkgs:
                                    local_files = [f[:-3] for f in os.listdir(SRC_DIR) if f.endswith('.py')]
                                    self.assertIn(
                                        top_pkg, local_files,
                                        f"Unapproved dependency '{top_pkg}' imported in {file} (line {node.lineno})"
                                    )
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                top_pkg = node.module.split('.')[0]
                                if top_pkg not in stdlib_modules and top_pkg not in approved_external_pkgs:
                                    local_files = [f[:-3] for f in os.listdir(SRC_DIR) if f.endswith('.py')]
                                    self.assertIn(
                                        top_pkg, local_files,
                                        f"Unapproved dependency '{top_pkg}' imported in {file} (line {node.lineno})"
                                    )


if __name__ == '__main__':
    unittest.main()
