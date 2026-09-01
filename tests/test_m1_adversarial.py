"""
M1 Adversarial Stress Test Suite
Comprehensive empirical tests exploring edge cases, invalid inputs,
malformed schemas, CLI combinations, and hardware diagnostic failure modes.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure src is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MAIN_SCRIPT = os.path.join(SRC_DIR, 'main.py')
CAMERA_TEST_SCRIPT = os.path.join(SRC_DIR, 'camera_test.py')

for d in [SRC_DIR, PROJECT_ROOT]:
    if d not in sys.path:
        sys.path.insert(0, d)

from data_preprocessing import GestureDataProcessor, parse_raw_sample
import camera_test


def make_valid_hand(scale: float = 1.0, offset=(0.0, 0.0, 0.0)):
    """Generate 21 valid landmark dicts."""
    lms = []
    for i in range(21):
        lms.append({
            'x': (0.1 * i) * scale + offset[0],
            'y': (0.05 * i) * scale + offset[1],
            'z': (-0.02 * i) * scale + offset[2],
            'visibility': 1.0
        })
    return lms


class TestDataPreprocessingAdversarial(unittest.TestCase):
    """Adversarial stress-testing of parse_raw_sample, normalize_landmarks, and prepare_dataset."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_adv_preproc_")
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.proc_dir = os.path.join(self.test_dir, "proc")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.proc_dir, exist_ok=True)
        self.processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.proc_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --- 1. parse_raw_sample Edge Cases ---
    def test_parse_empty_and_none(self):
        self.assertEqual(parse_raw_sample([]), [])
        self.assertEqual(parse_raw_sample(None), [])

    def test_parse_empty_nested_lists(self):
        self.assertEqual(parse_raw_sample([[]]), [])
        self.assertEqual(parse_raw_sample([[], []]), [])

    def test_parse_single_flat_hand(self):
        hand = make_valid_hand()
        parsed = parse_raw_sample(hand)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(parsed[0]), 21)

    def test_parse_nested_single_hand(self):
        hand = make_valid_hand()
        parsed = parse_raw_sample([hand])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(parsed[0]), 21)

    def test_parse_nested_multi_hands(self):
        hand1 = make_valid_hand(scale=1.0)
        hand2 = make_valid_hand(scale=0.8)
        hand3 = make_valid_hand(scale=0.5)
        parsed = parse_raw_sample([hand1, hand2, hand3])
        self.assertEqual(len(parsed), 3)
        self.assertEqual(len(parsed[0]), 21)
        self.assertEqual(len(parsed[1]), 21)
        self.assertEqual(len(parsed[2]), 21)

    def test_parse_single_empty_dict(self):
        # Malformed sample: [{}]
        parsed = parse_raw_sample([{}])
        self.assertEqual(parsed, [[{}]])

    # --- 2. normalize_landmarks Extreme & Malformed Coordinates ---
    def test_normalize_all_zeros(self):
        all_zeros = [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
        norm = self.processor.normalize_landmarks(all_zeros)
        self.assertEqual(len(norm), 21)
        for pt in norm:
            self.assertEqual(pt['x'], 0.0)
            self.assertEqual(pt['y'], 0.0)
            self.assertEqual(pt['z'], 0.0)

    def test_normalize_extreme_coordinates(self):
        extreme_offset = make_valid_hand(scale=1e6, offset=(1e8, -1e8, 1e7))
        norm = self.processor.normalize_landmarks(extreme_offset)
        self.assertEqual(len(norm), 21)
        for pt in norm:
            self.assertFalse(np.isnan(pt['x']))
            self.assertFalse(np.isinf(pt['x']))
            self.assertFalse(np.isnan(pt['y']))
            self.assertFalse(np.isnan(pt['z']))

    def test_normalize_tiny_coordinates(self):
        tiny = make_valid_hand(scale=1e-12, offset=(1e-10, 1e-10, 1e-10))
        norm = self.processor.normalize_landmarks(tiny)
        self.assertEqual(len(norm), 21)
        for pt in norm:
            self.assertFalse(np.isnan(pt['x']))

    def test_normalize_missing_visibility_key(self):
        hand = [{'x': 0.1 * i, 'y': 0.2 * i, 'z': 0.01 * i} for i in range(21)]
        norm = self.processor.normalize_landmarks(hand)
        self.assertEqual(len(norm), 21)
        for pt in norm:
            self.assertEqual(pt['visibility'], 1.0)

    def test_normalize_missing_coordinate_key_raises_keyerror(self):
        # Missing 'z' coordinate
        hand_no_z = [{'x': 0.1 * i, 'y': 0.2 * i} for i in range(21)]
        with self.assertRaises(KeyError):
            self.processor.normalize_landmarks(hand_no_z)

    def test_normalize_short_landmark_list_raises_indexerror(self):
        # Single landmark dict instead of 21 landmarks
        short_hand = [{'x': 0.1, 'y': 0.2, 'z': 0.3}]
        with self.assertRaises(IndexError):
            self.processor.normalize_landmarks(short_hand)

    # --- 3. prepare_dataset with Mixed 1-Hand / 2-Hand Datasets ---
    def test_mixed_one_hand_and_two_hand_dataset_padding(self):
        """Test dataset containing a mix of single-hand and two-hand samples across gestures."""
        # Gesture 1: Single-hand recorded
        g1_samples = [make_valid_hand(scale=1.0) for _ in range(10)]
        # Gesture 2: Two-hands recorded (nested 2 hands)
        g2_samples = [[make_valid_hand(scale=0.9), make_valid_hand(scale=1.1)] for _ in range(10)]
        # Gesture 3: Mixed samples in same gesture (some 1-hand, some 2-hands)
        g3_samples = []
        for i in range(10):
            if i % 2 == 0:
                g3_samples.append([make_valid_hand(scale=0.8)])
            else:
                g3_samples.append([make_valid_hand(scale=0.85), make_valid_hand(scale=1.05)])

        # Save files
        with open(os.path.join(self.raw_dir, "gesture1.json"), "w") as f:
            json.dump({'gesture_name': 'one_hand_g1', 'landmarks': g1_samples, 'two_hands': False}, f)
        with open(os.path.join(self.raw_dir, "gesture2.json"), "w") as f:
            json.dump({'gesture_name': 'two_hand_g2', 'landmarks': g2_samples, 'two_hands': True}, f)
        with open(os.path.join(self.raw_dir, "gesture3.json"), "w") as f:
            json.dump({'gesture_name': 'mixed_g3', 'landmarks': g3_samples, 'two_hands': True}, f)

        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.proc_dir)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=True)

        # Invariant checks
        self.assertEqual(len(class_names), 3)
        # All feature dimensions must be 126
        self.assertEqual(X_train.shape[1], 126)
        self.assertEqual(X_val.shape[1], 126)
        self.assertEqual(X_test.shape[1], 126)
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_val).any())
        self.assertFalse(np.isnan(X_test).any())

        # Verify load_processed_data loads it back accurately
        l_X_train, l_y_train, l_X_val, l_y_val, l_X_test, l_y_test, l_class_names, is_two_handed = processor.load_processed_data()
        self.assertTrue(is_two_handed)
        self.assertEqual(l_X_train.shape[1], 126)


class TestCLIAdversarial(unittest.TestCase):
    """Adversarial stress-testing of main.py CLI arguments."""

    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = [sys.executable, MAIN_SCRIPT] + args
        return subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

    def test_invalid_global_flags(self):
        result = self.run_cli(['--nonexistent-flag'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr.lower())

    def test_invalid_subcommand(self):
        result = self.run_cli(['unsupported_action'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())

    def test_collect_missing_required_flag(self):
        result = self.run_cli(['collect'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())

    def test_preprocess_nonexistent_input_dir(self):
        result = self.run_cli(['preprocess', '--input', 'non_existent_raw_dir_xyz_987'])
        # Main catches ValueError/FileNotFoundError and prints clean error
        self.assertEqual(result.returncode, 0)
        self.assertIn("Error during preprocessing", result.stdout)

    def test_train_nonexistent_data_dir(self):
        result = self.run_cli(['train', '--data', 'non_existent_proc_dir_xyz_987'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("Error: Processed data not found", result.stdout)

    def test_evaluate_nonexistent_data_dir(self):
        result = self.run_cli(['evaluate', '--data', 'non_existent_proc_dir_xyz_987'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("Error:", result.stdout)

    def test_train_invalid_model_type_choice(self):
        result = self.run_cli(['train', '--model-type', 'bert'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())


class TestCameraAdversarial(unittest.TestCase):
    """Adversarial testing of camera_test.py with edge case arguments and hardware fallbacks."""

    @patch('cv2.VideoCapture')
    def test_invalid_camera_index_returns_false(self, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        success = camera_test.test_camera(camera_id=99, headless=True)
        self.assertFalse(success)
        mock_videocapture.assert_called_once_with(99)

    @patch('cv2.VideoCapture')
    def test_camera_zero_duration_with_max_frames(self, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        # When duration is 0.0, max_frames=2 terminates the loop
        success = camera_test.test_camera(camera_id=0, duration=0.0, headless=True, max_frames=2)
        self.assertTrue(success)
        self.assertEqual(mock_cap.read.call_count, 3)  # 1 initial test + 2 loop frames

    @patch('cv2.VideoCapture')
    def test_camera_negative_duration_with_max_frames(self, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        # When duration is negative, max_frames=3 terminates the loop
        success = camera_test.test_camera(camera_id=0, duration=-10.0, headless=True, max_frames=3)
        self.assertTrue(success)
        self.assertEqual(mock_cap.read.call_count, 4)  # 1 initial + 3 loop frames

    @patch('cv2.destroyAllWindows')
    @patch('cv2.imshow')
    @patch('cv2.VideoCapture')
    def test_headless_mode_never_calls_imshow(self, mock_videocapture, mock_imshow, mock_destroy):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        success = camera_test.test_camera(camera_id=0, duration=0.1, headless=True, max_frames=2)
        self.assertTrue(success)
        mock_imshow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
