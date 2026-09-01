"""
Data Preprocessing & Schema Normalization Test Suite
Tests landmark normalization invariants, coordinate math, zero divisions, flattening,
augmentation, schema parsing, and NPZ dataset persistence.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import GestureDataProcessor


def make_dummy_landmarks(num_points: int = 21, base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), scale: float = 1.0) -> list[dict]:
    """Helper to generate a realistic hand landmark dictionary set."""
    landmarks = []
    # Point 0: Wrist
    landmarks.append({'x': 0.5 * scale + base_offset[0], 'y': 0.8 * scale + base_offset[1], 'z': 0.0 * scale + base_offset[2], 'visibility': 1.0})
    # Points 1-4: Thumb
    for i in range(1, 5):
        landmarks.append({'x': (0.4 - i * 0.05) * scale + base_offset[0], 'y': (0.7 - i * 0.05) * scale + base_offset[1], 'z': (-i * 0.02) * scale + base_offset[2], 'visibility': 1.0})
    # Points 5-8: Index
    for i in range(5, 9):
        landmarks.append({'x': (0.45 + (i-5) * 0.02) * scale + base_offset[0], 'y': (0.5 - (i-5) * 0.08) * scale + base_offset[1], 'z': (-(i-5) * 0.01) * scale + base_offset[2], 'visibility': 1.0})
    # Points 9-12: Middle (Point 9 is Middle MCP)
    # Point 9: Middle finger MCP
    landmarks.append({'x': 0.5 * scale + base_offset[0], 'y': 0.4 * scale + base_offset[1], 'z': 0.0 * scale + base_offset[2], 'visibility': 1.0})
    for i in range(10, 13):
        landmarks.append({'x': 0.5 * scale + base_offset[0], 'y': (0.4 - (i-9) * 0.08) * scale + base_offset[1], 'z': (-(i-9) * 0.01) * scale + base_offset[2], 'visibility': 1.0})
    # Points 13-16: Ring
    for i in range(13, 17):
        landmarks.append({'x': (0.55 - (i-13) * 0.01) * scale + base_offset[0], 'y': (0.45 - (i-13) * 0.07) * scale + base_offset[1], 'z': (-(i-13) * 0.01) * scale + base_offset[2], 'visibility': 1.0})
    # Points 17-20: Pinky
    for i in range(17, 21):
        landmarks.append({'x': (0.6 - (i-17) * 0.01) * scale + base_offset[0], 'y': (0.5 - (i-17) * 0.06) * scale + base_offset[1], 'z': (-(i-17) * 0.01) * scale + base_offset[2], 'visibility': 1.0})
    return landmarks


class TestLandmarkNormalization(unittest.TestCase):
    """Tier 1: Fast invariant and mathematical property tests for landmark normalization."""

    def setUp(self):
        self.processor = GestureDataProcessor(data_dir=tempfile.gettempdir(), processed_dir=tempfile.gettempdir())
        self.raw_landmarks = make_dummy_landmarks()

    def test_normalization_structure_and_length(self):
        """Test normalized output retains exactly 21 landmarks with x, y, z, visibility keys."""
        normalized = self.processor.normalize_landmarks(self.raw_landmarks)
        self.assertEqual(len(normalized), 21)
        for pt in normalized:
            self.assertIn('x', pt)
            self.assertIn('y', pt)
            self.assertIn('z', pt)
            self.assertIn('visibility', pt)
            self.assertFalse(np.isnan(pt['x']))
            self.assertFalse(np.isnan(pt['y']))
            self.assertFalse(np.isnan(pt['z']))

    def test_palm_center_is_origin(self):
        """Invariant: The palm center (midpoint of wrist and middle MCP) must map to (0, 0, 0)."""
        normalized = self.processor.normalize_landmarks(self.raw_landmarks)
        wrist = np.array([normalized[0]['x'], normalized[0]['y'], normalized[0]['z']])
        middle_mcp = np.array([normalized[9]['x'], normalized[9]['y'], normalized[9]['z']])
        palm_center = (wrist + middle_mcp) / 2.0
        np.testing.assert_allclose(palm_center, np.zeros(3), atol=1e-6)

    def test_unit_scale_reference_distance(self):
        """Invariant: The distance from wrist (0) to middle MCP (9) must normalize to exactly 1.0."""
        normalized = self.processor.normalize_landmarks(self.raw_landmarks)
        wrist = np.array([normalized[0]['x'], normalized[0]['y'], normalized[0]['z']])
        middle_mcp = np.array([normalized[9]['x'], normalized[9]['y'], normalized[9]['z']])
        dist = np.linalg.norm(middle_mcp - wrist)
        self.assertAlmostEqual(dist, 1.0, places=5)

    def test_translation_invariance(self):
        """Invariant: Adding a translation offset to raw landmarks produces identical normalized coordinates."""
        offset_landmarks = make_dummy_landmarks(base_offset=(15.5, -42.3, 8.8))
        norm_original = self.processor.normalize_landmarks(self.raw_landmarks)
        norm_offset = self.processor.normalize_landmarks(offset_landmarks)

        for pt_orig, pt_off in zip(norm_original, norm_offset):
            self.assertAlmostEqual(pt_orig['x'], pt_off['x'], places=5)
            self.assertAlmostEqual(pt_orig['y'], pt_off['y'], places=5)
            self.assertAlmostEqual(pt_orig['z'], pt_off['z'], places=5)

    def test_scale_invariance(self):
        """Invariant: Scaling raw landmarks by an arbitrary constant factor produces identical normalized coordinates."""
        scaled_landmarks = make_dummy_landmarks(scale=3.75)
        norm_original = self.processor.normalize_landmarks(self.raw_landmarks)
        norm_scaled = self.processor.normalize_landmarks(scaled_landmarks)

        for pt_orig, pt_sc in zip(norm_original, norm_scaled):
            self.assertAlmostEqual(pt_orig['x'], pt_sc['x'], places=5)
            self.assertAlmostEqual(pt_orig['y'], pt_sc['y'], places=5)
            self.assertAlmostEqual(pt_orig['z'], pt_sc['z'], places=5)

    def test_degenerate_zero_distance_no_zerodivision(self):
        """Boundary/Adversarial: Hand with identical wrist and middle MCP does not crash with ZeroDivisionError."""
        degenerate_landmarks = make_dummy_landmarks()
        # Set middle MCP equal to wrist
        degenerate_landmarks[9] = dict(degenerate_landmarks[0])
        
        normalized = self.processor.normalize_landmarks(degenerate_landmarks)
        self.assertEqual(len(normalized), 21)
        for pt in normalized:
            self.assertFalse(np.isnan(pt['x']))
            self.assertFalse(np.isinf(pt['x']))

    def test_all_zeros_hand(self):
        """Boundary/Adversarial: All zero coordinates do not cause exceptions and return zero-centered output."""
        all_zeros = [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
        normalized = self.processor.normalize_landmarks(all_zeros)
        self.assertEqual(len(normalized), 21)
        for pt in normalized:
            self.assertEqual(pt['x'], 0.0)
            self.assertEqual(pt['y'], 0.0)
            self.assertEqual(pt['z'], 0.0)


class TestLandmarkFlatteningAndAugmentation(unittest.TestCase):
    """Tier 1: Tests landmark vector flattening and augmentation invariants."""

    def setUp(self):
        self.processor = GestureDataProcessor(data_dir=tempfile.gettempdir(), processed_dir=tempfile.gettempdir())
        self.landmarks = make_dummy_landmarks()

    def test_flatten_dimensions(self):
        """Test flattening 21 3D landmarks produces a 63-element 1D numpy float array."""
        flat = self.processor.flatten_landmarks(self.landmarks)
        self.assertIsInstance(flat, np.ndarray)
        self.assertEqual(flat.shape, (63,))
        self.assertEqual(flat[0], self.landmarks[0]['x'])
        self.assertEqual(flat[1], self.landmarks[0]['y'])
        self.assertEqual(flat[2], self.landmarks[0]['z'])
        self.assertEqual(flat[60], self.landmarks[20]['x'])
        self.assertEqual(flat[61], self.landmarks[20]['y'])
        self.assertEqual(flat[62], self.landmarks[20]['z'])

    def test_augment_landmarks_count_and_shape(self):
        """Test augmentation generates requested number of sets each with 21 landmarks."""
        num_aug = 7
        augmented = self.processor.augment_landmarks(self.landmarks, num_augmentations=num_aug)
        self.assertEqual(len(augmented), num_aug)
        for aug_set in augmented:
            self.assertEqual(len(aug_set), 21)
            for pt in aug_set:
                self.assertIn('x', pt)
                self.assertIn('y', pt)
                self.assertIn('z', pt)
                self.assertFalse(np.isnan(pt['x']))

    def test_augmentation_variance(self):
        """Test augmentation produces non-identical perturbation from the original landmarks."""
        augmented = self.processor.augment_landmarks(self.landmarks, num_augmentations=3)
        flat_orig = self.processor.flatten_landmarks(self.landmarks)
        for aug_set in augmented:
            flat_aug = self.processor.flatten_landmarks(aug_set)
            # Should not be identical
            self.assertFalse(np.allclose(flat_orig, flat_aug, atol=1e-7))
            # But should remain reasonably bounded
            self.assertTrue(np.all(np.abs(flat_aug - flat_orig) < 2.0))


class TestSchemaParserAndDatasetPreparation(unittest.TestCase):
    """Tier 4: Pipeline integration tests for raw sample schemas and NPZ dataset persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_test_preproc_")
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Generate synthetic single-hand raw JSON files
        for gesture in ['hello', 'yes', 'no']:
            samples = []
            for _ in range(15):
                samples.append(make_dummy_landmarks())
            data = {
                'gesture_name': gesture,
                'timestamp': '20260901_120000',
                'num_samples': len(samples),
                'landmarks': samples,
                'two_hands': False
            }
            filepath = os.path.join(self.raw_dir, f"{gesture}_20260901.json")
            with open(filepath, 'w') as f:
                json.dump(data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_gesture_data_single_hand(self):
        """Test loading single-hand gesture JSON dataset."""
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir)
        gesture_data, is_two_handed = processor.load_gesture_data()
        self.assertFalse(is_two_handed)
        self.assertEqual(len(gesture_data), 3)
        self.assertIn('hello', gesture_data)
        self.assertIn('yes', gesture_data)
        self.assertIn('no', gesture_data)
        self.assertEqual(len(gesture_data['hello']), 15)

    def test_prepare_and_load_dataset_roundtrip(self):
        """Test complete dataset preparation, stratification, and NPZ round-trip loading."""
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir, random_seed=42)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=True)

        self.assertEqual(len(class_names), 3)
        self.assertEqual(X_train.shape[1], 63)
        self.assertEqual(X_val.shape[1], 63)
        self.assertEqual(X_test.shape[1], 63)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))

        # Check saved files exist
        npz_path = os.path.join(self.processed_dir, 'processed_gesture_data.npz')
        json_path = os.path.join(self.processed_dir, 'class_names.json')
        self.assertTrue(os.path.exists(npz_path))
        self.assertTrue(os.path.exists(json_path))

        # Load back
        loaded = processor.load_processed_data()
        l_X_train, l_y_train, l_X_val, l_y_val, l_X_test, l_y_test, l_class_names, l_two_handed = loaded
        np.testing.assert_array_equal(X_train, l_X_train)
        np.testing.assert_array_equal(y_train, l_y_train)
        self.assertEqual(class_names, l_class_names)
        self.assertFalse(l_two_handed)

    def test_two_handed_dataset_parsing(self):
        """Test dataset preparation with two-handed format (nested 1-hand or 2-hands samples)."""
        two_hand_raw = os.path.join(self.test_dir, "two_hand_raw")
        two_hand_proc = os.path.join(self.test_dir, "two_hand_proc")
        os.makedirs(two_hand_raw, exist_ok=True)
        os.makedirs(two_hand_proc, exist_ok=True)

        for gesture in ['clap', 'wave']:
            samples = []
            for i in range(12):
                if i % 2 == 0:
                    # 2 hands detected
                    samples.append([make_dummy_landmarks(scale=0.9), make_dummy_landmarks(scale=1.1)])
                else:
                    # 1 hand detected in 2-hand mode
                    samples.append([make_dummy_landmarks(scale=1.0)])
            data = {
                'gesture_name': gesture,
                'timestamp': '20260901_120000',
                'num_samples': len(samples),
                'landmarks': samples,
                'two_hands': True
            }
            with open(os.path.join(two_hand_raw, f"{gesture}.json"), 'w') as f:
                json.dump(data, f)

        processor = GestureDataProcessor(data_dir=two_hand_raw, processed_dir=two_hand_proc)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=False)
        # Feature dimension must be 126 for two-handed models
        self.assertEqual(X_train.shape[1], 126)
        self.assertEqual(len(class_names), 2)


if __name__ == '__main__':
    unittest.main()
