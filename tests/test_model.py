"""
Model Architecture & Tensor Inference Test Suite
Tests Dense MLP architecture, direct tensor execution, softmax probability conservation,
model persistence, metadata serialization, and prediction contracts.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import tensorflow as tf

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_training import GestureModelTrainer


class TestModelArchitectureAndTensors(unittest.TestCase):
    """Tier 2: Model architecture shapes, loss functions, and direct tensor inference."""

    def setUp(self):
        self.trainer = GestureModelTrainer(model_dir=tempfile.gettempdir(), random_seed=42)

    def test_single_hand_dense_model_layers_and_shapes(self):
        """Test Dense MLP architecture for single-hand input (63 dimensions)."""
        num_classes = 4
        model = self.trainer.build_model(input_shape=(63,), num_classes=num_classes, is_two_handed=False)

        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 63))
        self.assertEqual(model.output_shape, (None, num_classes))
        self.assertEqual(model.loss, 'sparse_categorical_crossentropy')

        # Verify layer types
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        self.assertIn('Dense', layer_names)
        self.assertIn('BatchNormalization', layer_names)
        self.assertIn('Dropout', layer_names)

        # Verify final layer activation is softmax
        self.assertEqual(model.layers[-1].activation.__name__, 'softmax')

    def test_two_handed_dense_model_shapes(self):
        """Test Dense MLP architecture for two-handed input (126 dimensions)."""
        num_classes = 6
        model = self.trainer.build_model(input_shape=(126,), num_classes=num_classes, is_two_handed=True)
        self.assertEqual(model.input_shape, (None, 126))
        self.assertEqual(model.output_shape, (None, num_classes))
        self.assertTrue(self.trainer.is_two_handed)

    def test_lstm_model_shapes(self):
        """Test LSTM model architecture shape and compile configuration."""
        num_classes = 3
        model = self.trainer.build_lstm_model(input_shape=(63,), num_classes=num_classes)
        self.assertEqual(model.input_shape, (None, 63))
        self.assertEqual(model.output_shape, (None, num_classes))

    def test_direct_tensor_inference_execution(self):
        """Invariant: Direct callable tensor execution model(x, training=False) yields valid probability distributions."""
        num_classes = 5
        model = self.trainer.build_model(input_shape=(63,), num_classes=num_classes)

        # Single sample inference
        dummy_sample = np.random.randn(1, 63).astype(np.float32)
        tensor_input = tf.constant(dummy_sample)

        # Direct tensor inference (avoids slow model.predict graph overhead in real-time loops)
        output_tensor = model(tensor_input, training=False)
        output_np = output_tensor.numpy()

        self.assertEqual(output_np.shape, (1, num_classes))
        self.assertTrue(np.all(output_np >= 0.0), "All output probabilities must be non-negative")
        self.assertTrue(np.all(output_np <= 1.0), "All output probabilities must be <= 1.0")
        self.assertAlmostEqual(float(np.sum(output_np[0])), 1.0, places=5)

    def test_batch_tensor_inference_probability_conservation(self):
        """Invariant: Softmax sum of probabilities equals 1.0 across all batch elements."""
        num_classes = 4
        model = self.trainer.build_model(input_shape=(63,), num_classes=num_classes)

        batch_size = 16
        batch_input = tf.constant(np.random.randn(batch_size, 63).astype(np.float32))
        batch_output = model(batch_input, training=False).numpy()

        self.assertEqual(batch_output.shape, (batch_size, num_classes))
        sums = np.sum(batch_output, axis=1)
        np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)

    def test_predict_proba_single_and_batch(self):
        """Invariant: predict_proba handles 1D vector (63,) and 2D batch (B, 63) returning conserved probabilities."""
        num_classes = 4
        self.trainer.build_model(input_shape=(63,), num_classes=num_classes)
        self.trainer.class_names = ['a', 'b', 'c', 'd']

        # Single 1D vector
        sample_1d = np.random.randn(63).astype(np.float32)
        probs_1d = self.trainer.predict_proba(sample_1d)
        self.assertEqual(probs_1d.shape, (num_classes,))
        self.assertAlmostEqual(float(np.sum(probs_1d)), 1.0, places=5)

        # Batch 2D array
        batch_2d = np.random.randn(8, 63).astype(np.float32)
        probs_2d = self.trainer.predict_proba(batch_2d)
        self.assertEqual(probs_2d.shape, (8, num_classes))
        np.testing.assert_allclose(np.sum(probs_2d, axis=1), np.ones(8), atol=1e-5)


class TestModelTrainingAndPersistence(unittest.TestCase):
    """Tier 4: Model training loop, save/load cycle, and metadata preservation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sl_test_model_")
        self.trainer = GestureModelTrainer(model_dir=self.test_dir, random_seed=42)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_save_load_predict_cycle(self):
        """Test fast end-to-end model training, serialization, loading, and prediction."""
        class_names = ['hello', 'thank_you', 'yes']
        num_classes = len(class_names)
        num_samples = 30

        np.random.seed(42)
        X_train = np.random.randn(num_samples, 63).astype(np.float32)
        y_train = np.random.randint(0, num_classes, size=num_samples)
        X_val = np.random.randn(10, 63).astype(np.float32)
        y_val = np.random.randint(0, num_classes, size=10)

        # Train for 2 epochs on mini dataset
        history = self.trainer.train(
            X_train, y_train, X_val, y_val, class_names,
            epochs=2, batch_size=8, model_type='dense', is_two_handed=False
        )

        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

        # Verify metadata file was written
        metadata_path = os.path.join(self.test_dir, 'model_metadata.json')
        self.assertTrue(os.path.exists(metadata_path))

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['model_type'], 'dense')
        self.assertEqual(metadata['class_names'], class_names)
        self.assertEqual(metadata['input_shape'], 63)
        self.assertEqual(metadata['num_classes'], num_classes)
        self.assertFalse(metadata['is_two_handed'])

        # Create new trainer instance and load saved model
        new_trainer = GestureModelTrainer(model_dir=self.test_dir)
        loaded_model = new_trainer.load_model()

        self.assertIsNotNone(loaded_model)
        self.assertEqual(new_trainer.class_names, class_names)
        self.assertEqual(new_trainer.input_shape, (63,))

        # Test single sample prediction contract
        test_sample = np.random.randn(63).astype(np.float32)
        pred_class, confidence = new_trainer.predict(test_sample)

        self.assertIn(pred_class, class_names)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
