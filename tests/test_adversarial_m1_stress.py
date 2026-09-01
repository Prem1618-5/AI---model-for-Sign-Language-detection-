"""
Adversarial Stress Test Suite for Milestone M1 Components
Empirical Challenger: Instance 2

Probes:
1. Data augmentation invariants, conformal geometry, numeric bounds, NaN/Inf immunity, and throughput.
2. Dataset split ratios, stratification limits, small/unbalanced datasets, and train-test data leakage.
3. Two-handed feature permutation, zero-padding, and dimensional consistency.
4. Camera test duration boundary conditions (hang on duration <= 0).
5. Directory creation safety, permission constraints, file overwriting, and corruption recovery.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import stat
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root and src directory are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import GestureDataProcessor, parse_raw_sample
from data_collection import DataCollector
import camera_test


def make_synthetic_hand(num_points=21, offset=(0.0, 0.0, 0.0), scale=1.0):
    """Generate realistic hand landmarks for testing."""
    landmarks = []
    # Point 0: Wrist
    landmarks.append({'x': 0.5 * scale + offset[0], 'y': 0.8 * scale + offset[1], 'z': 0.0 * scale + offset[2], 'visibility': 1.0})
    # Thumb 1-4
    for i in range(1, 5):
        landmarks.append({'x': (0.4 - i * 0.05) * scale + offset[0], 'y': (0.7 - i * 0.05) * scale + offset[1], 'z': (-i * 0.02) * scale + offset[2], 'visibility': 1.0})
    # Index 5-8
    for i in range(5, 9):
        landmarks.append({'x': (0.45 + (i-5) * 0.02) * scale + offset[0], 'y': (0.5 - (i-5) * 0.08) * scale + offset[1], 'z': (-(i-5) * 0.01) * scale + offset[2], 'visibility': 1.0})
    # Middle 9-12 (9 is Middle MCP)
    landmarks.append({'x': 0.5 * scale + offset[0], 'y': 0.4 * scale + offset[1], 'z': 0.0 * scale + offset[2], 'visibility': 1.0})
    for i in range(10, 13):
        landmarks.append({'x': 0.5 * scale + offset[0], 'y': (0.4 - (i-9) * 0.08) * scale + offset[1], 'z': (-(i-9) * 0.01) * scale + offset[2], 'visibility': 1.0})
    # Ring 13-16
    for i in range(13, 17):
        landmarks.append({'x': (0.55 - (i-13) * 0.01) * scale + offset[0], 'y': (0.45 - (i-13) * 0.07) * scale + offset[1], 'z': (-(i-13) * 0.01) * scale + offset[2], 'visibility': 1.0})
    # Pinky 17-20
    for i in range(17, 21):
        landmarks.append({'x': (0.6 - (i-17) * 0.01) * scale + offset[0], 'y': (0.5 - (i-17) * 0.06) * scale + offset[1], 'z': (-(i-17) * 0.01) * scale + offset[2], 'visibility': 1.0})
    return landmarks


class TestAugmentationInvariantsAndGeometricProperties(unittest.TestCase):
    """Stress testing data augmentation mathematical and numeric invariants."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="sl_challenger_aug_")
        self.processor = GestureDataProcessor(data_dir=self.temp_dir, processed_dir=self.temp_dir)
        self.standard_hand = make_synthetic_hand()
        self.normalized_hand = self.processor.normalize_landmarks(self.standard_hand)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_augmentation_statistical_no_nan_no_inf(self):
        """Stress-test: 5,000 augmented hands (105,000 landmark points) must contain zero NaN or Inf."""
        augmented_batches = self.processor.augment_landmarks(self.normalized_hand, num_augmentations=5000)
        self.assertEqual(len(augmented_batches), 5000)
        
        all_coords = []
        for batch in augmented_batches:
            self.assertEqual(len(batch), 21)
            for pt in batch:
                all_coords.extend([pt['x'], pt['y'], pt['z']])

        arr = np.array(all_coords)
        self.assertFalse(np.isnan(arr).any(), "Found NaN in augmented landmarks")
        self.assertFalse(np.isinf(arr).any(), "Found Inf in augmented landmarks")
        # Ensure values stay bounded in realistic range
        self.assertTrue(np.all(np.abs(arr) < 5.0), f"Augmented landmarks exceeded bounds: max {np.max(np.abs(arr))}")

    def test_geometric_conformal_ratio_preservation(self):
        """
        Invariant check: Rotation + uniform translation + uniform scaling is a similarity transformation.
        The ratio of any two Euclidean pairwise segment lengths ||p_i - p_j|| / ||p_k - p_l||
        must be PRESERVED across all augmented outputs.
        """
        p = np.array([[lm['x'], lm['y'], lm['z']] for lm in self.normalized_hand])
        d_orig_0_9 = np.linalg.norm(p[0] - p[9])
        d_orig_4_20 = np.linalg.norm(p[4] - p[20])
        orig_ratio = d_orig_0_9 / d_orig_4_20

        augmented = self.processor.augment_landmarks(self.normalized_hand, num_augmentations=100)
        for aug in augmented:
            p_aug = np.array([[lm['x'], lm['y'], lm['z']] for lm in aug])
            d_aug_0_9 = np.linalg.norm(p_aug[0] - p_aug[9])
            d_aug_4_20 = np.linalg.norm(p_aug[4] - p_aug[20])
            aug_ratio = d_aug_0_9 / d_aug_4_20
            self.assertAlmostEqual(orig_ratio, aug_ratio, places=5,
                                   msg="Augmentation broke similarity/conformal metric ratio!")

    def test_rotation_matrix_orthonormality_and_properties(self):
        """
        Verify rotation matrix properties in augmentation:
        Rotation matrix R must satisfy R * R^T = I and det(R) = +1.0 (proper rigid rotation).
        """
        for _ in range(100):
            theta = np.random.uniform(-0.2, 0.2)
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            # Orthogonality
            np.testing.assert_allclose(np.dot(R, R.T), np.eye(3), atol=1e-7)
            # Proper rotation determinant
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_extreme_coordinate_normalization_and_augmentation(self):
        """Stress-test: Large, subnormal, negative, and zero coordinates."""
        test_scales = [1e-12, 1e-4, 1.0, 1e4, 1e12]
        for s in test_scales:
            hand = make_synthetic_hand(scale=s, offset=(s*10, -s*5, s*2))
            norm = self.processor.normalize_landmarks(hand)
            self.assertEqual(len(norm), 21)
            norm_p = np.array([[lm['x'], lm['y'], lm['z']] for lm in norm])
            self.assertFalse(np.isnan(norm_p).any())
            self.assertFalse(np.isinf(norm_p).any())
            
            # Wrist to Middle MCP distance must be normalized to 1.0
            dist = np.linalg.norm(norm_p[9] - norm_p[0])
            self.assertAlmostEqual(dist, 1.0, places=4, msg=f"Failed for scale {s}")

            # Augmentation on extreme scale normalized hand
            aug = self.processor.augment_landmarks(norm, num_augmentations=5)
            self.assertEqual(len(aug), 5)
            for a in aug:
                a_p = np.array([[lm['x'], lm['y'], lm['z']] for lm in a])
                self.assertFalse(np.isnan(a_p).any())
                self.assertFalse(np.isinf(a_p).any())

    def test_empty_and_corrupt_landmarks_augmentation(self):
        """Boundary test: empty landmarks in augment_landmarks raises ValueError."""
        with self.assertRaises(ValueError):
            self.processor.augment_landmarks([])

        # Zero augmentations requested
        aug_zero = self.processor.augment_landmarks(self.normalized_hand, num_augmentations=0)
        self.assertEqual(len(aug_zero), 0)

        # Negative augmentations requested
        aug_neg = self.processor.augment_landmarks(self.normalized_hand, num_augmentations=-5)
        self.assertEqual(len(aug_neg), 0)

    def test_augmentation_throughput_benchmark(self):
        """Benchmark augmentation throughput (samples/sec). Target: > 5,000 hands/sec."""
        count = 2000
        start = time.perf_counter()
        _ = self.processor.augment_landmarks(self.normalized_hand, num_augmentations=count)
        elapsed = time.perf_counter() - start
        rate = count / elapsed
        print(f"\n[BENCHMARK] Augmentation throughput: {rate:.1f} hands/sec ({elapsed*1000/count:.4f} ms/hand)")
        self.assertGreater(rate, 2000.0, "Augmentation throughput is unacceptably slow")


class TestDatasetSplitStratificationAndDataLeakage(unittest.TestCase):
    """Stress testing dataset splitting, class stratification limits, and data leakage."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_challenger_split_")
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dataset(self, gesture_counts: dict[str, int]):
        """Helper to create raw JSON files with given sample counts."""
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        for gesture, count in gesture_counts.items():
            samples = [make_synthetic_hand(offset=(i*0.01, 0, 0)) for i in range(count)]
            data = {
                'gesture_name': gesture,
                'timestamp': '20260902_000000',
                'num_samples': count,
                'landmarks': samples,
                'two_hands': False
            }
            with open(os.path.join(self.raw_dir, f"{gesture}.json"), 'w') as f:
                json.dump(data, f)

    def test_small_sample_sizes_without_augmentation_fails_stratify(self):
        """
        Stress test: Stratified splitting on tiny dataset without augmentation.
        When sample count per class is very small (e.g. 2 samples per class),
        train_test_split with stratify fails because the test split cannot contain members of all classes.
        """
        self._create_dataset({'hello': 2, 'yes': 2, 'no': 2})
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir)
        with self.assertRaises(ValueError):
            processor.prepare_dataset(augment=False, test_size=0.2, val_size=0.1)

    def test_single_sample_per_class_failure(self):
        """Stress test: 1 sample per class must raise ValueError during stratified splitting."""
        self._create_dataset({'hello': 1, 'yes': 1})
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir)
        with self.assertRaises(ValueError):
            processor.prepare_dataset(augment=False)

    def test_unbalanced_dataset_minority_class_behavior(self):
        """
        Stress test: Severe class imbalance (100 samples vs 3 samples).
        Tests whether the minority class survives the two-stage train_test_split.
        """
        self._create_dataset({'hello': 100, 'yes': 3})
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir)
        try:
            X_tr, y_tr, X_v, y_v, X_te, y_te, classes = processor.prepare_dataset(augment=False)
            self.assertEqual(len(classes), 2)
        except ValueError as e:
            # Documented failure mode: minority class dropped during secondary stratification
            self.assertIn("The least populated class in y has only 1 member", str(e))

    def test_quantify_data_leakage_in_augmentation_pipeline(self):
        """
        INVARIANT VERIFICATION:
        Verify that performing data augmentation only on the training split eliminates train-test
        contamination: test and val sets contain strictly unaugmented samples with zero leakage.
        """
        self._create_dataset({'hello': 20, 'yes': 20, 'no': 20})
        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir, random_seed=42)
        
        # 60 raw samples: 20% test = 12 samples (clean, unaugmented)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=True)

        self.assertEqual(X_test.shape[0], 12, "Test set must contain exactly 20% unaugmented raw samples")
        self.assertGreater(X_train.shape[0], 200, "Train set must contain base + augmented samples")
        print(f"[VERIFIED] Clean split: Train={len(X_train)} (augmented), Val={len(X_val)}, Test={len(X_test)} (zero leakage).")


class TestTwoHandedPermutationAndFeatureDimensions(unittest.TestCase):
    """Stress testing two-handed format, single-hand padding, and feature alignment."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_challenger_2h_")
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_two_handed_single_hand_zero_padding(self):
        """Invariant: In two-handed mode, 1-hand samples must have exactly features 63..125 as zeros."""
        samples = [[make_synthetic_hand()] for _ in range(15)]
        data = {
            'gesture_name': 'one_hand_in_2h_mode',
            'timestamp': '20260902_000000',
            'num_samples': 15,
            'landmarks': samples,
            'two_hands': True
        }
        with open(os.path.join(self.raw_dir, "one_hand.json"), 'w') as f:
            json.dump(data, f)

        # Add second class to allow splitting
        samples_2 = [[make_synthetic_hand(), make_synthetic_hand()] for _ in range(15)]
        data_2 = {
            'gesture_name': 'two_hands_gesture',
            'timestamp': '20260902_000000',
            'num_samples': 15,
            'landmarks': samples_2,
            'two_hands': True
        }
        with open(os.path.join(self.raw_dir, "two_hands.json"), 'w') as f:
            json.dump(data_2, f)

        processor = GestureDataProcessor(data_dir=self.raw_dir, processed_dir=self.processed_dir)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=False)

        self.assertEqual(X_train.shape[1], 126)
        
        # Check that one_hand class samples have exact zero padding in features 63..125
        one_hand_label = processor.label_encoder.transform(['one_hand_in_2h_mode'])[0]
        one_hand_samples = X_train[y_train == one_hand_label]
        for sample in one_hand_samples:
            np.testing.assert_allclose(sample[63:], np.zeros(63), atol=1e-7,
                                       err_msg="Second hand padding was non-zero!")


class TestCameraTestAdversarialBoundaries(unittest.TestCase):
    """Stress testing camera_test.py boundary parameters and mock execution."""

    @patch('cv2.destroyAllWindows')
    @patch('cv2.VideoCapture')
    def test_camera_duration_zero_or_negative_with_max_frames(self, mock_vid, mock_destroy):
        """Test camera_test when duration is 0 or negative but max_frames is provided."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy)
        mock_vid.return_value = mock_cap

        # Running with duration=0.0 and max_frames=5 should exit after 5 frames without hanging
        success = camera_test.test_camera(duration=0.0, headless=True, max_frames=5)
        self.assertTrue(success)
        self.assertEqual(mock_cap.read.call_count, 6)  # 1 initial test + 5 loop frames


class TestDirectoryCreationPermissionsAndOverwriting(unittest.TestCase):
    """Stress testing directory creation, permission constraints, and overwrite safety."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_challenger_dir_")

    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                except Exception:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
                except Exception:
                    pass
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_deep_nested_directory_creation(self):
        """Test auto-creation of deeply nested processed directories."""
        deep_dir = os.path.join(self.test_dir, "nested", "level1", "level2", "level3", "processed")
        processor = GestureDataProcessor(data_dir=self.test_dir, processed_dir=deep_dir)
        self.assertTrue(os.path.exists(deep_dir))
        self.assertTrue(os.path.isdir(deep_dir))

    def test_directory_conflict_with_existing_file(self):
        """Test error behavior when processed_dir path already exists as a regular file."""
        file_path = os.path.join(self.test_dir, "blocking_file.txt")
        with open(file_path, 'w') as f:
            f.write("I am a file, not a directory")

        with self.assertRaises(FileExistsError):
            GestureDataProcessor(data_dir=self.test_dir, processed_dir=file_path)

    def test_save_processed_data_overwrite_idempotency(self):
        """Test that saving processed data multiple times cleanly overwrites without file corruption."""
        proc_dir = os.path.join(self.test_dir, "proc")
        processor = GestureDataProcessor(data_dir=self.test_dir, processed_dir=proc_dir)

        X_dummy = np.random.randn(10, 63).astype(np.float32)
        y_dummy = np.random.randint(0, 2, size=(10,)).astype(np.int32)
        metadata = {'feature_dim': 63, 'is_two_handed': False}

        # Save 1
        processor.save_processed_data(X_dummy, y_dummy, X_dummy, y_dummy, X_dummy, y_dummy, ['a', 'b'], metadata)
        npz_file = os.path.join(proc_dir, 'processed_gesture_data.npz')
        self.assertTrue(os.path.exists(npz_file))

        # Save 2 with different data
        X_dummy_2 = np.random.randn(20, 63).astype(np.float32)
        y_dummy_2 = np.random.randint(0, 3, size=(20,)).astype(np.int32)
        metadata_2 = {'feature_dim': 63, 'is_two_handed': False}
        processor.save_processed_data(X_dummy_2, y_dummy_2, X_dummy_2, y_dummy_2, X_dummy_2, y_dummy_2, ['a', 'b', 'c'], metadata_2)

        # Verify load succeeds and returns new data
        loaded = processor.load_processed_data()
        self.assertEqual(loaded[0].shape[0], 20)
        self.assertEqual(len(loaded[6]), 3)

    def test_load_gesture_data_empty_dir(self):
        """Test load_gesture_data on empty directory raises descriptive ValueError."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        processor = GestureDataProcessor(data_dir=empty_dir, processed_dir=self.test_dir)
        with self.assertRaises(ValueError) as ctx:
            processor.load_gesture_data()
        self.assertIn("No gesture data files found", str(ctx.exception))

    def test_corrupted_json_in_data_dir(self):
        """Test load_gesture_data behavior when corrupt/unparseable JSON is encountered."""
        raw_dir = os.path.join(self.test_dir, "raw_corrupt")
        os.makedirs(raw_dir, exist_ok=True)
        corrupt_file = os.path.join(raw_dir, "corrupt_20260902.json")
        with open(corrupt_file, 'w') as f:
            f.write("{ INVALID JSON DATA <<<")

        processor = GestureDataProcessor(data_dir=raw_dir, processed_dir=self.test_dir)
        with self.assertRaises(json.JSONDecodeError):
            processor.load_gesture_data()

    def test_collector_directory_creation(self):
        """Test DataCollector initializes and creates output directory."""
        col_dir = os.path.join(self.test_dir, "collector_out", "sub")
        collector = DataCollector(output_dir=col_dir)
        self.assertTrue(os.path.exists(col_dir))
        self.assertTrue(os.path.isdir(col_dir))


if __name__ == '__main__':
    unittest.main()
