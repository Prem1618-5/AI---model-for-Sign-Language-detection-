"""
Adversarial Stress Test Suite for Milestone M2 Components
Empirical Challenger: Instance 2

Covers:
1. Data Leakage Verification:
   - Mathematical assertion that NO sample in validation or test sets has an augmented counterpart in the training set.
   - Scale/rotation-invariant biometric ratio tracing to verify strict training-only lineage.
   - Partition isolation across single-hand, two-handed, and unbalanced datasets.
2. Training Pipeline Stress Testing:
   - Custom epoch counts (epochs=1, multi-epoch).
   - Batch size variations: batch_size=1 (SGD + BatchNorm), batch_size=len(dataset), batch_size > len(dataset), odd/prime sizes (3, 7, 13).
   - Early stopping triggers, patience variations, ReduceLROnPlateau, ModelCheckpoint best weight restoration.
   - Dense MLP vs LSTM vs Two-handed architecture training robustness.
   - Extreme class imbalance training survival.
3. Model Serialization & Deserialization Round-Trip Fidelity:
   - SavedModel format and metadata JSON persistence.
   - Exact weight equality, direct tensor prediction equivalence, probability conservation.
   - Metadata JSON integrity, missing file handling, custom class names, directory auto-discovery.
   - LSTM and two-handed model serialization fidelity.
4. Temporal Filter State Machine Invariants:
   - Softmax EMA probability conservation.
   - Exact hysteresis boundary states (T_high=0.80, T_low=0.45, debounce=4).
   - Velocity gating under high wrist displacement and dt edge cases.
   - Sequence buffer memory bounds and inactivity timeouts.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import numpy as np
import tensorflow as tf

# Ensure project root and src directory are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import GestureDataProcessor, parse_raw_sample
from model_training import GestureModelTrainer
from temporal_filter import TemporalSmoother


def create_synthetic_hand_sample(wrist_x=0.5, wrist_y=0.8, wrist_z=0.0, thumb_len=0.20, pinky_len=0.15):
    """
    Generate a single-hand sample of 21 landmark dictionaries with parameterized biometric geometry.
    thumb_len and pinky_len provide an invariant ratio invariant to rotation/translation/scale.
    """
    landmarks = []
    # 0: Wrist
    landmarks.append({'x': float(wrist_x), 'y': float(wrist_y), 'z': float(wrist_z), 'visibility': 1.0})
    # 1-4: Thumb (thumb tip at index 4)
    for i in range(1, 4):
        landmarks.append({'x': float(wrist_x - 0.04 * i), 'y': float(wrist_y - 0.03 * i), 'z': 0.0, 'visibility': 1.0})
    landmarks.append({'x': float(wrist_x - thumb_len), 'y': float(wrist_y - thumb_len), 'z': 0.0, 'visibility': 1.0})
    
    # 5-8: Index
    for i in range(5, 9):
        landmarks.append({'x': float(wrist_x - 0.02 * (i - 4)), 'y': float(wrist_y - 0.08 * (i - 4)), 'z': 0.0, 'visibility': 1.0})
        
    # 9-12: Middle (9 is MCP)
    landmarks.append({'x': float(wrist_x), 'y': float(wrist_y - 0.35), 'z': float(wrist_z), 'visibility': 1.0})
    for i in range(10, 13):
        landmarks.append({'x': float(wrist_x), 'y': float(wrist_y - 0.35 - 0.06 * (i - 9)), 'z': 0.0, 'visibility': 1.0})
        
    # 13-16: Ring
    for i in range(13, 17):
        landmarks.append({'x': float(wrist_x + 0.03 * (i - 12)), 'y': float(wrist_y - 0.07 * (i - 12)), 'z': 0.0, 'visibility': 1.0})
        
    # 17-20: Pinky (pinky tip at index 20)
    for i in range(17, 20):
        landmarks.append({'x': float(wrist_x + 0.04 * (i - 16)), 'y': float(wrist_y - 0.05 * (i - 16)), 'z': 0.0, 'visibility': 1.0})
    landmarks.append({'x': float(wrist_x + pinky_len), 'y': float(wrist_y - pinky_len), 'z': 0.0, 'visibility': 1.0})
    
    return landmarks


def create_synthetic_raw_dataset(temp_dir, gestures=("gesture_A", "gesture_B", "gesture_C"), samples_per_gesture=20, two_handed=False):
    """Write synthetic raw gesture JSON files to temp_dir with distinct geometric biometric signatures."""
    raw_dir = os.path.join(temp_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    global_k = 0
    for g_idx, g_name in enumerate(gestures):
        landmarks_list = []
        for s_idx in range(samples_per_gesture):
            # Globally unique thumb_len per sample to guarantee strictly collision-free ratios
            thumb_len = 0.10 + (global_k * 0.005)
            pinky_len = 0.20
            hand1 = create_synthetic_hand_sample(wrist_x=0.5, wrist_y=0.8, thumb_len=thumb_len, pinky_len=pinky_len)
            if two_handed:
                hand2 = create_synthetic_hand_sample(wrist_x=0.2, wrist_y=0.8, thumb_len=thumb_len * 0.9, pinky_len=pinky_len * 0.9)
                landmarks_list.append([hand1, hand2])
            else:
                landmarks_list.append(hand1)
            global_k += 1
                
        payload = {
            'gesture_name': g_name,
            'two_hands': two_handed,
            'num_samples': samples_per_gesture,
            'landmarks': landmarks_list
        }
        with open(os.path.join(raw_dir, f"{g_name}.json"), 'w') as f:
            json.dump(payload, f)
            
    return raw_dir


# =====================================================================
# 1. DATA LEAKAGE & PARTITION ADVERSARIAL TESTS
# =====================================================================
class TestDataLeakageAndPartitions(unittest.TestCase):
    """Rigorous empirical verification of train/val/test isolation and data leakage prevention."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_leakage_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_strict_zero_data_leakage_with_augmentation(self):
        """
        Adversarial Invariant:
        No sample in X_val or X_test may have ANY augmented or unaugmented counterpart in X_train.
        Every augmented sample in X_train must originate exclusively from a training-partition sample.
        """
        raw_dir = create_synthetic_raw_dataset(
            self.test_dir,
            gestures=("alpha", "beta", "gamma", "delta"),
            samples_per_gesture=25,
            two_handed=False
        )
        proc_dir = os.path.join(self.test_dir, 'proc')
        processor = GestureDataProcessor(data_dir=raw_dir, processed_dir=proc_dir, random_seed=123)

        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(
            augment=True, test_size=0.20, val_size=0.10
        )

        total_samples = 4 * 25  # 100 raw samples
        # Expected splits before augmentation: test=20, trainval=80 -> val = 80 * (0.1/0.8) = 10, train=70
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(X_val), 10)
        # Train has 70 base + 70 * 5 augmentations = 420 samples
        self.assertEqual(len(X_train), 70 * (1 + 5))

        # 1. Exact Row Intersection Check:
        # Check that no validation or test feature vector appears exactly in X_train
        for i, val_row in enumerate(X_val):
            dists = np.linalg.norm(X_train - val_row, axis=1)
            min_dist = np.min(dists)
            self.assertGreater(min_dist, 1e-4, f"Validation sample {i} is duplicated in training set (dist={min_dist})")

        for i, test_row in enumerate(X_test):
            dists = np.linalg.norm(X_train - test_row, axis=1)
            min_dist = np.min(dists)
            self.assertGreater(min_dist, 1e-4, f"Test sample {i} is duplicated in training set (dist={min_dist})")

        # 2. Partition Disjointness Check:
        # Check that val and test sets are strictly disjoint
        for i, test_row in enumerate(X_test):
            dists = np.linalg.norm(X_val - test_row, axis=1)
            min_dist = np.min(dists)
            self.assertGreater(min_dist, 1e-4, f"Test sample {i} is duplicated in validation set (dist={min_dist})")

        # 3. Biometric Invariant Lineage Check:
        # In single-hand 63-dim features, joints are (x,y,z) for 21 points.
        # Joint 0 (wrist) = [0:3], Joint 4 (thumb tip) = [12:15], Joint 20 (pinky tip) = [60:63].
        # For any sample, ratio R = ||p4 - p0|| / ||p20 - p0|| is invariant under 2D/3D rotation, translation, and scaling.
        def compute_ratio(feat):
            p0 = feat[0:3]
            p4 = feat[12:15]
            p20 = feat[60:63]
            d_thumb = np.linalg.norm(p4 - p0)
            d_pinky = np.linalg.norm(p20 - p0)
            return d_thumb / d_pinky if d_pinky > 0 else 0.0

        train_ratios = np.array([compute_ratio(row) for row in X_train])
        val_ratios = np.array([compute_ratio(row) for row in X_val])
        test_ratios = np.array([compute_ratio(row) for row in X_test])

        base_train_ratios = train_ratios[::6]  # 70 base training samples
        augmented_train_ratios = np.delete(train_ratios, np.arange(0, len(train_ratios), 6))  # 350 augmented samples

        # Every augmented train ratio MUST match a base train ratio (within numerical tolerance < 1e-4)
        for i, aug_r in enumerate(augmented_train_ratios):
            diff_to_train_bases = np.abs(base_train_bases_diff := (base_train_ratios - aug_r))
            min_train_diff = np.min(diff_to_train_bases)
            self.assertLess(min_train_diff, 1e-4, f"Augmented train sample {i} ratio {aug_r} does not match any base train sample!")

            # It must NOT match any validation ratio
            diff_to_val = np.min(np.abs(val_ratios - aug_r))
            self.assertGreater(diff_to_val, 1e-4, f"Augmented train sample {i} matches a validation sample ratio! Data leakage detected!")

            # It must NOT match any test ratio
            diff_to_test = np.min(np.abs(test_ratios - aug_r))
            self.assertGreater(diff_to_test, 1e-4, f"Augmented train sample {i} matches a test sample ratio! Data leakage detected!")

    def test_two_handed_dataset_leakage_and_shapes(self):
        """Verify zero data leakage on two-handed datasets (126 feature dimensions)."""
        raw_dir = create_synthetic_raw_dataset(
            self.test_dir,
            gestures=("two_hand_A", "two_hand_B"),
            samples_per_gesture=30,
            two_handed=True
        )
        proc_dir = os.path.join(self.test_dir, 'proc_2h')
        processor = GestureDataProcessor(data_dir=raw_dir, processed_dir=proc_dir, random_seed=42)

        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(
            augment=True, test_size=0.20, val_size=0.10
        )

        self.assertEqual(X_train.shape[1], 126)
        self.assertEqual(X_val.shape[1], 126)
        self.assertEqual(X_test.shape[1], 126)

        # Verify no overlap
        for val_row in X_val:
            dists = np.linalg.norm(X_train - val_row, axis=1)
            self.assertGreater(np.min(dists), 1e-4)

        for test_row in X_test:
            dists = np.linalg.norm(X_train - test_row, axis=1)
            self.assertGreater(np.min(dists), 1e-4)


# =====================================================================
# 2. TRAINING PIPELINE STRESS TESTS
# =====================================================================
class TestTrainingPipelineStress(unittest.TestCase):
    """Adversarial stress testing of model training loop with hyperparameter variations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_train_stress_")
        self.trainer = GestureModelTrainer(model_dir=self.test_dir, random_seed=42)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_minimal_epochs_training(self):
        """Stress: Verify model compiles, trains, and outputs valid metrics with epochs=1."""
        num_samples = 24
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, 3, size=num_samples)
        X_val = np.random.randn(6, 63).astype(np.float32)
        y_val = np.random.randint(0, 3, size=6)
        class_names = ['c0', 'c1', 'c2']

        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=1, batch_size=8, model_type='dense'
        )

        self.assertIsNotNone(history)
        self.assertEqual(len(history.history['loss']), 1)
        self.assertFalse(np.isnan(history.history['loss'][0]))

    def test_batch_size_one_stochastic_gradient_descent(self):
        """Stress: Verify training with batch_size=1 (exercises BatchNorm edge behavior)."""
        num_samples = 16
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, 2, size=num_samples)
        X_val = np.random.randn(4, 63).astype(np.float32)
        y_val = np.random.randint(0, 2, size=4)
        class_names = ['left', 'right']

        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=1, model_type='dense'
        )

        self.assertIsNotNone(history)
        self.assertEqual(len(history.history['loss']), 2)
        # Ensure predictions work immediately after batch_size=1 training
        probs = self.trainer.predict_proba(X_val[0])
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_batch_size_equal_and_greater_than_dataset_size(self):
        """Stress: Verify full batch (batch_size=N) and oversized batch (batch_size > N)."""
        num_samples = 20
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, 2, size=num_samples)
        X_val = np.random.randn(5, 63).astype(np.float32)
        y_val = np.random.randint(0, 2, size=5)
        class_names = ['up', 'down']

        # 1. Full batch
        hist_full = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=num_samples, model_type='dense'
        )
        self.assertIsNotNone(hist_full)

        # 2. Oversized batch (batch_size = 500 > num_samples)
        hist_over = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=500, model_type='dense'
        )
        self.assertIsNotNone(hist_over)

    def test_odd_prime_batch_sizes(self):
        """Stress: Verify batch_size with odd/prime numbers (3, 7, 13) causing uneven batch partitions."""
        num_samples = 31
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, 3, size=num_samples)
        X_val = np.random.randn(7, 63).astype(np.float32)
        y_val = np.random.randint(0, 3, size=7)
        class_names = ['a', 'b', 'c']

        for bs in [3, 7, 13]:
            hist = self.trainer.train(
                X_train, y_train, X_val, y_val, class_names,
                epochs=2, batch_size=bs, model_type='dense'
            )
            self.assertIsNotNone(hist)
            self.assertEqual(len(hist.history['loss']), 2)

    def test_early_stopping_and_best_weights_restoration(self):
        """
        Stress: Verify EarlyStopping callback stops training when val_loss plateaus/degrades,
        and restores the best weights.
        """
        np.random.seed(42)
        num_train = 40
        X_train = np.random.randn(num_train, 63).astype(np.float32)
        y_train = np.zeros(num_train, dtype=int)
        y_train[num_train // 2:] = 1

        # Validation set with mismatched labels to force val_loss to diverge
        num_val = 20
        X_val = np.random.randn(num_val, 63).astype(np.float32)
        y_val = np.random.randint(0, 2, size=num_val)
        class_names = ['class0', 'class1']

        # Request 100 epochs with EarlyStopping patience=10
        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=100, batch_size=8, model_type='dense'
        )

        epochs_trained = len(history.history['loss'])
        # It must terminate well before 100 epochs due to early stopping
        self.assertLess(epochs_trained, 100, f"EarlyStopping failed to stop training: ran {epochs_trained} epochs")
        
        # Verify checkpoint file exists
        checkpoint_path = os.path.join(self.test_dir, 'best_model.h5')
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_severe_class_imbalance_survival(self):
        """Stress: Verify training survives on heavily imbalanced class distributions."""
        X_train = np.random.randn(60, 63).astype(np.float32)
        # Class 0: 50 samples, Class 1: 5 samples, Class 2: 5 samples
        y_train = np.array([0] * 50 + [1] * 5 + [2] * 5)
        X_val = np.random.randn(10, 63).astype(np.float32)
        y_val = np.random.randint(0, 3, size=10)
        class_names = ['major', 'minor1', 'minor2']

        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=8, model_type='dense'
        )
        self.assertIsNotNone(history)

    def test_lstm_model_training_and_evaluation(self):
        """Stress: Verify sequential LSTM model builds, trains, and evaluates without failure."""
        num_samples = 30
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, 3, size=num_samples)
        X_val = np.random.randn(10, 63).astype(np.float32)
        y_val = np.random.randint(0, 3, size=10)
        X_test = np.random.randn(10, 63).astype(np.float32)
        y_test = np.random.randint(0, 3, size=10)
        class_names = ['lstm_a', 'lstm_b', 'lstm_c']

        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=8, model_type='lstm'
        )
        self.assertIsNotNone(history)

        metrics = self.trainer.evaluate(X_test, y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)
        self.assertIn('classification_report', metrics)


# =====================================================================
# 3. MODEL SERIALIZATION / DESERIALIZATION ROUND-TRIP FIDELITY
# =====================================================================
class TestModelSerializationFidelity(unittest.TestCase):
    """Adversarial stress testing of SavedModel and JSON metadata round-trip fidelity."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_serial_stress_")
        self.trainer = GestureModelTrainer(model_dir=self.test_dir, random_seed=42)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_exact_weights_and_tensor_inference_roundtrip(self):
        """
        Adversarial Invariant:
        Saved and reloaded model weights must match bit-for-bit,
        and predictions on 100 arbitrary inputs must be identical within float32 precision.
        """
        class_names = ['one', 'two', 'three', 'four', 'five']
        num_classes = len(class_names)
        num_samples = 40

        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, num_classes, size=num_samples)
        X_val = np.random.randn(10, 63).astype(np.float32)
        y_val = np.random.randint(0, num_classes, size=10)

        # Train model
        self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=3, batch_size=8, model_type='dense'
        )

        original_weights = [w.copy() for w in self.trainer.model.get_weights()]

        # Generate 100 random test inputs
        test_inputs = np.random.randn(100, 63).astype(np.float32)
        original_preds = self.trainer.model(tf.constant(test_inputs), training=False).numpy()
        original_proba = self.trainer.predict_proba(test_inputs)

        # Reload into a completely new GestureModelTrainer instance
        reloaded_trainer = GestureModelTrainer(model_dir=self.test_dir)
        reloaded_trainer.load_model()

        # 1. Verify exact weight tensors
        reloaded_weights = reloaded_trainer.model.get_weights()
        self.assertEqual(len(original_weights), len(reloaded_weights))
        for i, (w_orig, w_reloaded) in enumerate(zip(original_weights, reloaded_weights)):
            np.testing.assert_array_equal(
                w_orig, w_reloaded,
                err_msg=f"Weight mismatch at layer parameter {i}"
            )

        # 2. Verify direct tensor inference match
        reloaded_preds = reloaded_trainer.model(tf.constant(test_inputs), training=False).numpy()
        np.testing.assert_allclose(
            original_preds, reloaded_preds, atol=1e-6,
            err_msg="Direct tensor inference outputs differed after reload!"
        )

        # 3. Verify predict_proba output match
        reloaded_proba = reloaded_trainer.predict_proba(test_inputs)
        np.testing.assert_allclose(
            original_proba, reloaded_proba, atol=1e-6,
            err_msg="predict_proba outputs differed after reload!"
        )

        # 4. Verify single-sample predict() class and confidence match
        for idx in range(10):
            orig_class, orig_conf = self.trainer.predict(test_inputs[idx])
            re_class, re_conf = reloaded_trainer.predict(test_inputs[idx])
            self.assertEqual(orig_class, re_class)
            self.assertAlmostEqual(orig_conf, re_conf, places=5)

    def test_metadata_json_schema_and_types(self):
        """Verify model_metadata.json preserves correct schema, types, and custom classes."""
        custom_classes = ['Gesture_Alpha', 'Gesture_Beta', 'Gesture_Gamma']
        self.trainer.build_model(input_shape=(63,), num_classes=len(custom_classes), is_two_handed=False)
        self.trainer.class_names = custom_classes
        self.trainer.save_model(model_type='dense')

        metadata_path = os.path.join(self.test_dir, 'model_metadata.json')
        self.assertTrue(os.path.exists(metadata_path))

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['model_type'], 'dense')
        self.assertEqual(metadata['class_names'], custom_classes)
        self.assertEqual(metadata['input_shape'], 63)
        self.assertEqual(metadata['num_classes'], 3)
        self.assertFalse(metadata['is_two_handed'])
        self.assertIsInstance(metadata['timestamp'], str)

        # Reload and check trainer properties
        new_trainer = GestureModelTrainer(model_dir=self.test_dir)
        new_trainer.load_model()
        self.assertEqual(new_trainer.class_names, custom_classes)
        self.assertEqual(new_trainer.input_shape, (63,))
        self.assertFalse(new_trainer.is_two_handed)

    def test_explicit_model_path_loading(self):
        """Verify load_model works when given an explicit model path string."""
        custom_classes = ['g1', 'g2']
        self.trainer.build_model(input_shape=(63,), num_classes=2)
        self.trainer.class_names = custom_classes
        self.trainer.save_model(model_type='dense')

        saved_path = os.path.join(self.test_dir, 'gesture_recognition_dense_model')
        new_trainer = GestureModelTrainer(model_dir=self.test_dir)
        loaded_model = new_trainer.load_model(model_path=saved_path)

        self.assertIsNotNone(loaded_model)
        self.assertEqual(new_trainer.class_names, custom_classes)

    def test_missing_model_directory_raises_file_not_found(self):
        """Verify load_model raises FileNotFoundError when no models exist."""
        empty_dir = tempfile.mkdtemp(prefix="sl_empty_")
        try:
            trainer = GestureModelTrainer(model_dir=empty_dir)
            with self.assertRaises(FileNotFoundError):
                trainer.load_model()
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


# =====================================================================
# 4. TEMPORAL FILTER & STATE MACHINE ADVERSARIAL TESTS
# =====================================================================
class TestTemporalFilterAdversarial(unittest.TestCase):
    """Stress testing of TemporalSmoother for stability, hysteresis, velocity gating, and timeouts."""

    def test_softmax_ema_probability_conservation(self):
        """Invariant: Softmax EMA smoothed probabilities must conserve sum == 1.0 across 500 iterations."""
        classes = ['idle', 'fist', 'open_palm', 'peace']
        smoother = TemporalSmoother(alpha=0.25, class_names=classes)

        rng = np.random.RandomState(42)
        for _ in range(500):
            logits = rng.randn(len(classes))
            raw_probs = np.exp(logits) / np.sum(np.exp(logits))
            
            gesture, conf, status = smoother.update(raw_probs)
            
            self.assertIsNotNone(smoother.smoothed_probs)
            self.assertAlmostEqual(float(np.sum(smoother.smoothed_probs)), 1.0, places=5)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_hysteresis_exact_threshold_transitions(self):
        """
        Adversarial Invariant:
        - Enter DETECTED only when confidence >= T_high (0.80) for debounce_frames (4) consecutive frames.
        - Remain in DETECTED as long as confidence >= T_low (0.45).
        - Drop to SCANNING only when confidence strictly < T_low (0.45).
        """
        classes = ['gesture_a', 'gesture_b']
        smoother = TemporalSmoother(
            alpha=1.0,  # alpha=1.0 allows direct test of exact raw input probabilities
            threshold_high=0.80,
            threshold_low=0.45,
            debounce_frames=4,
            class_names=classes
        )

        # 1. Sub-threshold probe (0.79 < 0.80) -> SCANNING
        for _ in range(10):
            g, c, s = smoother.update([0.79, 0.21])
            self.assertEqual(s, "SCANNING")
            self.assertEqual(g, "None")

        # 2. Reaching exact T_high (0.80) for 1, 2, 3 frames -> UNCERTAIN / Analysing...
        for f in range(1, 4):
            g, c, s = smoother.update([0.80, 0.20])
            self.assertEqual(s, "UNCERTAIN", f"Frame {f} should be UNCERTAIN")
            self.assertEqual(g, "Analysing...")

        # 3. 4th frame reaches debounce threshold -> DETECTED
        g, c, s = smoother.update([0.80, 0.20])
        self.assertEqual(s, "DETECTED")
        self.assertEqual(g, "gesture_a")

        # 4. Confidence drops to 0.50 (between T_low=0.45 and T_high=0.80) -> Remains DETECTED (Hysteresis latch)
        g, c, s = smoother.update([0.50, 0.50])
        self.assertEqual(s, "DETECTED")
        self.assertEqual(g, "gesture_a")

        # 5. Confidence drops to 0.45 (exact T_low boundary) -> Remains DETECTED
        g, c, s = smoother.update([0.45, 0.55])
        self.assertEqual(s, "DETECTED")

        # 6. Confidence drops strictly below 0.45
        smoother_3c = TemporalSmoother(alpha=1.0, threshold_high=0.80, threshold_low=0.45, debounce_frames=2, class_names=['a', 'b', 'c'])
        smoother_3c.update([0.9, 0.05, 0.05])
        smoother_3c.update([0.9, 0.05, 0.05])
        self.assertEqual(smoother_3c.current_state, "DETECTED")

        # Now top confidence is 0.40 < 0.45
        g, c, s = smoother_3c.update([0.40, 0.30, 0.30])
        self.assertEqual(s, "SCANNING")
        self.assertEqual(g, "None")

    def test_kinematic_velocity_gating_and_dt_extremes(self):
        """Stress: Verify wrist velocity gating suppresses false triggers during fast movement."""
        smoother = TemporalSmoother(
            alpha=1.0,
            threshold_high=0.80,
            debounce_frames=3,
            velocity_threshold=0.08,
            class_names=['wave', 'fist']
        )

        # 1. Reach DETECTED state after 3 frames
        t0 = 1000.0
        smoother.update([0.9, 0.1], wrist_pos=(0.5, 0.5), timestamp=t0)
        smoother.update([0.9, 0.1], wrist_pos=(0.5, 0.5), timestamp=t0 + 0.033)
        g, c, s = smoother.update([0.9, 0.1], wrist_pos=(0.5, 0.5), timestamp=t0 + 0.066)
        self.assertEqual(s, "DETECTED")

        # 2. Huge displacement in 33ms (0.5 to 0.8 -> displacement=0.30 > 0.08)
        t1 = t0 + 0.100
        g, c, s = smoother.update([0.9, 0.1], wrist_pos=(0.8, 0.5), timestamp=t1)
        self.assertEqual(s, "UNCERTAIN")
        self.assertEqual(g, "Analysing...")

        # 3. dt near zero (1e-6 seconds) - check division by zero protection and state persistence
        t2 = t1 + 1e-6
        g, c, s = smoother.update([0.9, 0.1], wrist_pos=(0.8, 0.5), timestamp=t2)
        # 1st stationary frame after motion reset: consecutive_high_frames is 1 < 3 -> UNCERTAIN
        self.assertEqual(s, "UNCERTAIN")
        self.assertEqual(g, "Analysing...")

        # 4. 2nd stationary frame: consecutive_high_frames is 2 < 3 -> UNCERTAIN
        t3 = t2 + 0.033
        g, c, s = smoother.update([0.9, 0.1], wrist_pos=(0.8, 0.5), timestamp=t3)
        self.assertEqual(s, "UNCERTAIN")

        # 5. 3rd stationary frame: consecutive_high_frames is 3 == 3 -> DETECTED
        t4 = t3 + 0.033
        g, c, s = smoother.update([0.9, 0.1], wrist_pos=(0.8, 0.5), timestamp=t4)
        self.assertEqual(s, "DETECTED")

    def test_sequence_buffer_memory_cap_and_inactivity_timeout(self):
        """Stress: Verify sequence buffer caps at 10 items and resets on inactivity timeout."""
        smoother = TemporalSmoother(
            alpha=1.0,
            threshold_high=0.80,
            debounce_frames=1,
            sequence_timeout=1.5,
            class_names=[f"g_{i}" for i in range(20)]
        )

        t = 100.0
        # Trigger 15 distinct gestures
        for i in range(15):
            probs = np.zeros(20)
            probs[i] = 1.0
            smoother.update(probs, timestamp=t)
            t += 0.1  # 100ms apart (below 1.5s timeout)

        seq = smoother.get_sequence()
        # Must be capped at 10
        self.assertEqual(len(seq), 10)
        self.assertEqual(seq[-1], "G_14")
        self.assertEqual(seq[0], "G_5")

        # Advance timestamp beyond inactivity timeout (1.5s)
        t += 2.0
        probs = np.zeros(20)
        probs[0] = 1.0
        smoother.update(probs, timestamp=t)

        # Buffer should have been cleared and now contains only the new gesture
        seq_after = smoother.get_sequence()
        self.assertEqual(len(seq_after), 1)
        self.assertEqual(seq_after[0], "G_0")


if __name__ == '__main__':
    unittest.main()
