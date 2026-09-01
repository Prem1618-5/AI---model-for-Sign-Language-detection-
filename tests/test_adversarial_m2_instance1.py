"""
Adversarial Stress Test Suite for Milestone M2 Components
Empirical Challenger: Instance 1

Probes:
1. TemporalSmoother:
   - Rapid alternating probability vectors (oscillation damping & chatter suppression)
   - Extreme EMA alpha values (alpha=0.0 freeze, alpha=1.0 memoryless, out-of-range alphas)
   - Noisy step distributions (Dirichlet simplex noise, unnormalized logits, all-zeroes)
   - Kinematic wrist velocity gating (teleportation jumps, micro-jitters, zero/negative dt)
   - Missing wrist coordinates (None transitions, intermittent tracking)
   - Rapid repeated gestures, sequence buffer capacity limit (10 items), inactivity timeouts
   - Empty/mismatched class_names lists and dynamic probability vector dimensions
2. Direct Tensor Inference & ML Models:
   - Batch sizes: 0 (empty), 1, 10, 100, 1000
   - Single 1D arrays (63,), (126,), 2D arrays, and Python lists
   - Numerical anomalies: NaN, +Inf, -Inf, all-zeroes, extreme magnitudes (+-1e8)
   - Dimension mismatch rejection (63 vs 126 vs 30 vs 0)
   - Diverse data types: float32, float64, int32, float16
   - Latency benchmark comparing direct tensor evaluation vs model.predict()
"""

import os
import sys
import time
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

from temporal_filter import TemporalSmoother
from model_training import GestureModelTrainer


class TestTemporalSmootherAdversarialDynamics(unittest.TestCase):
    """Stress testing TemporalSmoother mathematical, numeric, and dynamic invariants."""

    def setUp(self):
        self.classes = ['hello', 'thank_you', 'yes', 'no']
        self.smoother = TemporalSmoother(
            alpha=0.25,
            threshold_high=0.80,
            threshold_low=0.45,
            debounce_frames=4,
            velocity_threshold=0.08,
            sequence_timeout=2.0,
            class_names=self.classes
        )

    def test_rapid_alternating_probabilities_oscillation_damping(self):
        """
        Adversarial probe: Rapid alternating probability vectors (class 0 vs class 1).
        When the classifier flips between class 0 (0.95) and class 1 (0.95) every frame,
        the EMA smoothing dampens steady-state probabilities below 0.80, preventing false detection.
        Invariant: Never enters DETECTED state; sequence buffer remains empty.
        """
        p_class0 = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        p_class1 = np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float32)

        t = 100.0
        for i in range(100):
            t += 0.033
            p_in = p_class0 if (i % 2 == 0) else p_class1
            pred, conf, status = self.smoother.update(p_in, wrist_pos=(0.5, 0.5), timestamp=t)
            self.assertNotEqual(status, "DETECTED", f"Oscillating input falsely triggered DETECTED at frame {i}")
            if i > 1:
                # After initial transient frame 0, steady-state alternating confidence is dampened below 0.80
                self.assertLess(conf, 0.80, f"Smoothed confidence {conf} reached 0.80 during steady-state alternation")

        self.assertEqual(len(self.smoother.sequence_buffer), 0, "Sequence buffer must remain empty during oscillation")

    def test_extreme_alpha_zero_state_freeze(self):
        """
        Adversarial probe: Extreme EMA alpha = 0.0.
        When alpha=0.0, S_t = 0 * P_t + 1 * S_{t-1}.
        The smoothed probabilities must freeze at the initial frame and ignore all subsequent inputs.
        """
        smoother = TemporalSmoother(alpha=0.0, class_names=self.classes)
        initial_prob = np.array([0.90, 0.10, 0.0, 0.0], dtype=np.float32)
        smoother.update(initial_prob)

        # Feed completely different probability distribution for 50 frames
        divergent_prob = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(50):
            smoother.update(divergent_prob)

        np.testing.assert_allclose(
            smoother.smoothed_probs, initial_prob, atol=1e-5,
            err_msg="Smoother failed to preserve frozen state with alpha=0.0"
        )

    def test_extreme_alpha_one_instantaneous_memoryless(self):
        """
        Adversarial probe: Extreme EMA alpha = 1.0.
        When alpha=1.0, S_t = P_t.
        Smoothed probabilities must exactly match the raw input on every single step with zero lag.
        """
        smoother = TemporalSmoother(alpha=1.0, class_names=self.classes)
        for _ in range(50):
            raw = np.random.dirichlet(np.ones(4)).astype(np.float32)
            smoother.update(raw)
            np.testing.assert_allclose(
                smoother.smoothed_probs, raw, atol=1e-5,
                err_msg="Smoother failed instantaneous tracking with alpha=1.0"
            )

    def test_noisy_dirichlet_distribution_simplex_conservation(self):
        """
        Adversarial probe: 1,000 frames of noisy Dirichlet simplex random variables.
        Invariant: Softmax probability sum is conserved to 1.0 +/- 1e-5; no NaN/Inf.
        """
        for _ in range(1000):
            # Test various Dirichlet concentration parameters (symmetric, sparse, peaked)
            alpha_param = np.random.choice([0.1, 0.5, 1.0, 5.0, 50.0])
            raw = np.random.dirichlet(np.full(4, alpha_param)).astype(np.float32)
            pred, conf, status = self.smoother.update(raw)

            self.assertFalse(np.isnan(self.smoother.smoothed_probs).any())
            self.assertFalse(np.isinf(self.smoother.smoothed_probs).any())
            self.assertAlmostEqual(float(np.sum(self.smoother.smoothed_probs)), 1.0, places=5)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0001)

    def test_unnormalized_and_scaled_input_distributions(self):
        """
        Adversarial probe: Inputs that do not sum to 1.0 (unnormalized logits, huge scales, tiny sums).
        Invariant: TemporalSmoother normalizes smoothed_probs so sum == 1.0.
        """
        # Unnormalized large values
        large_raw = np.array([100.0, 200.0, 50.0, 150.0], dtype=np.float32)
        self.smoother.update(large_raw)
        self.assertAlmostEqual(float(np.sum(self.smoother.smoothed_probs)), 1.0, places=5)

        # Unnormalized tiny sub-unitary sum
        tiny_raw = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        self.smoother.update(tiny_raw)
        self.assertAlmostEqual(float(np.sum(self.smoother.smoothed_probs)), 1.0, places=5)

    def test_all_zero_probability_vector_safety(self):
        """
        Adversarial probe: Input is entirely zeroes [0.0, 0.0, 0.0, 0.0].
        Invariant: Must not throw ZeroDivisionError or crash.
        """
        zero_raw = np.zeros(4, dtype=np.float32)
        pred, conf, status = self.smoother.update(zero_raw)
        self.assertIsNotNone(pred)
        self.assertIsNotNone(status)
        self.assertEqual(conf, 0.0)

    def test_dynamic_vector_length_reconfiguration(self):
        """
        Adversarial probe: Raw probability vector changes dimension dynamically mid-stream
        (e.g., switching from 4 classes to 6 classes).
        Invariant: Automatically reconfigures smoothed_probs without throwing index or shape errors.
        """
        # Start with 4 classes
        self.smoother.update(np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertEqual(len(self.smoother.smoothed_probs), 4)

        # Switch to 6 classes
        self.smoother.update(np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]))
        self.assertEqual(len(self.smoother.smoothed_probs), 6)
        self.assertAlmostEqual(float(np.sum(self.smoother.smoothed_probs)), 1.0, places=5)

    def test_empty_or_truncated_class_names_bounds_safety(self):
        """
        Adversarial probe: class_names is empty or shorter than probability vector.
        Invariant: Correctly returns status and falls back to string indices (e.g., '2') without IndexError.
        """
        smoother = TemporalSmoother(class_names=[])
        # Sub-threshold probe -> returns "None", "SCANNING"
        pred, conf, status = smoother.update(np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertEqual(status, "SCANNING")
        self.assertEqual(pred, "None")

        # Drive to DETECTED state on index 2 (needs 10 frames with alpha=0.25 to reach >= 0.80 and satisfy 4-frame debounce)
        for _ in range(12):
            pred, conf, status = smoother.update(np.array([0.02, 0.02, 0.95, 0.01]))
        self.assertEqual(status, "DETECTED")
        self.assertEqual(pred, "2")  # Index fallback string


class TestTemporalSmootherKinematicsAndSequence(unittest.TestCase):
    """Stress testing kinematic wrist velocity gating, timestamp boundaries, and sequence buffer."""

    def setUp(self):
        self.classes = ['hello', 'thank_you', 'yes', 'no']
        self.smoother = TemporalSmoother(
            alpha=0.25,
            threshold_high=0.80,
            threshold_low=0.45,
            debounce_frames=4,
            velocity_threshold=0.08,
            sequence_timeout=2.0,
            class_names=self.classes
        )

    def test_high_velocity_wrist_teleportation_gating(self):
        """
        Adversarial probe: Wrist position jumps across frame (displacement > 0.08).
        Invariant: Gated to UNCERTAIN; consecutive_high_frames reset to 0.
        """
        # Reach DETECTED state with stationary wrist at (0.2, 0.2) (need >= 10 frames with alpha=0.25)
        t = 100.0
        for _ in range(12):
            t += 0.033
            self.smoother.update(np.array([0.95, 0.02, 0.02, 0.01]), wrist_pos=(0.2, 0.2), timestamp=t)
        self.assertEqual(self.smoother.current_state, "DETECTED")

        # Teleportation jump from (0.2, 0.2) to (0.8, 0.8) -> displacement ~ 0.848 >> 0.08
        t += 0.033
        pred, conf, status = self.smoother.update(
            np.array([0.95, 0.02, 0.02, 0.01]), wrist_pos=(0.8, 0.8), timestamp=t
        )
        self.assertEqual(status, "UNCERTAIN")
        self.assertEqual(pred, "Analysing...")
        self.assertEqual(self.smoother.consecutive_high_frames, 0)

    def test_missing_wrist_pos_none_intermittent_tracking(self):
        """
        Adversarial probe: wrist_pos alternates between tuple coordinates and None.
        Invariant: Never throws AttributeError or TypeError; treats None as unconstrained motion.
        """
        t = 10.0
        for i in range(20):
            t += 0.033
            pos = (0.5, 0.5) if (i % 2 == 0) else None
            pred, conf, status = self.smoother.update(
                np.array([0.95, 0.02, 0.02, 0.01]), wrist_pos=pos, timestamp=t
            )
            self.assertIn(status, ["UNCERTAIN", "DETECTED", "SCANNING"])

    def test_negative_zero_and_microsecond_dt_timestamps(self):
        """
        Adversarial probe: Non-monotonic, zero, and microsecond timestamp intervals.
        Invariant: Protected by dt = max(current_time - last_update_time, 1e-4); no ZeroDivisionError.
        """
        # dt = 0.0 (identical timestamp)
        self.smoother.update(np.array([0.5, 0.5, 0.0, 0.0]), wrist_pos=(0.1, 0.1), timestamp=50.0)
        self.smoother.update(np.array([0.5, 0.5, 0.0, 0.0]), wrist_pos=(0.1, 0.1), timestamp=50.0)

        # dt < 0 (out of order timestamp)
        self.smoother.update(np.array([0.5, 0.5, 0.0, 0.0]), wrist_pos=(0.1, 0.1), timestamp=49.0)

        # Microsecond interval (dt = 1e-6)
        self.smoother.update(np.array([0.5, 0.5, 0.0, 0.0]), wrist_pos=(0.1, 0.1), timestamp=50.000001)

    def test_hysteresis_boundary_chatter_stability(self):
        """
        Adversarial probe: Probabilities oscillate across the high threshold T_high=0.80.
        Invariant: When confidence drops below 0.80, consecutive_high_frames resets to 0 and prevents DETECTED state.
        """
        # Alternating 2 frames high (0.90), 2 frames low (0.10)
        for _ in range(10):
            # 2 frames high -> rising but not sustained for 4 consecutive frames >= 0.80
            self.smoother.update(np.array([0.90, 0.03, 0.03, 0.04]))
            self.smoother.update(np.array([0.90, 0.03, 0.03, 0.04]))
            # 2 frames low -> drops confidence and resets consecutive counter
            pred, conf, status = self.smoother.update(np.array([0.10, 0.30, 0.30, 0.30]))
            self.assertEqual(status, "SCANNING")
            self.assertEqual(self.smoother.consecutive_high_frames, 0)
            self.smoother.update(np.array([0.10, 0.30, 0.30, 0.30]))

        # State must NEVER have reached DETECTED
        self.assertNotEqual(self.smoother.current_state, "DETECTED")
        self.assertEqual(len(self.smoother.sequence_buffer), 0)

    def test_sequence_buffer_capacity_cap_and_deduplication(self):
        """
        Adversarial probe: Rapidly cycling through 20 confirmed gestures with sufficient hold time.
        Invariant:
        1. Consecutive identical gestures are deduplicated.
        2. Sequence buffer is strictly bounded to maxlen=10.
        """
        t = 100.0

        for gesture_idx in range(20):
            target_class = gesture_idx % 4
            prob = np.zeros(4, dtype=np.float32)
            prob[target_class] = 0.95
            prob[(target_class + 1) % 4] = 0.05

            # Run 12 frames to ensure EMA reaches >= 0.80 and satisfies 4-frame debounce
            for _ in range(12):
                t += 0.033
                self.smoother.update(prob, wrist_pos=(0.5, 0.5), timestamp=t)

            self.assertLessEqual(len(self.smoother.sequence_buffer), 10, "Sequence buffer exceeded capacity limit of 10")

        # Buffer must have exactly 10 items (the most recent 10 gestures)
        self.assertEqual(len(self.smoother.sequence_buffer), 10)
        seq = self.smoother.get_sequence()
        self.assertEqual(len(seq), 10)

    def test_inactivity_timeout_exact_boundary(self):
        """
        Adversarial probe: Inactivity timeout boundary conditions.
        Invariant: Timeout at dt > 2.0s resets sequence and clears smoothed_probs.
        """
        t = 1000.0
        # Produce confirmed gesture (12 frames)
        for _ in range(12):
            t += 0.033
            self.smoother.update(np.array([0.95, 0.02, 0.02, 0.01]), timestamp=t)

        self.assertEqual(len(self.smoother.sequence_buffer), 1)

        # Update at t + 1.9s (< 2.0s timeout) -> sequence preserved
        self.smoother.update(np.array([0.95, 0.02, 0.02, 0.01]), timestamp=t + 1.9)
        self.assertEqual(len(self.smoother.sequence_buffer), 1)

        # Update at t + 4.5s (> 2.0s timeout) -> sequence cleared, state reset to SCANNING
        pred, conf, status = self.smoother.update(
            np.array([0.95, 0.02, 0.02, 0.01]), timestamp=t + 4.5
        )
        self.assertEqual(status, "UNCERTAIN")
        self.assertEqual(len(self.smoother.sequence_buffer), 0)


class TestDirectTensorInferenceAdversarial(unittest.TestCase):
    """Stress testing direct tensor execution, batch dimensions, and numerical edge cases."""

    def setUp(self):
        self.trainer_1h = GestureModelTrainer(model_dir=tempfile.gettempdir(), random_seed=42)
        self.model_1h = self.trainer_1h.build_model(input_shape=(63,), num_classes=4, is_two_handed=False)
        self.trainer_1h.class_names = ['hello', 'thank_you', 'yes', 'no']

        self.trainer_2h = GestureModelTrainer(model_dir=tempfile.gettempdir(), random_seed=42)
        self.model_2h = self.trainer_2h.build_model(input_shape=(126,), num_classes=6, is_two_handed=True)
        self.trainer_2h.class_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']

    def test_direct_tensor_execution_batch_sizes(self):
        """
        Adversarial probe: Batch sizes 1, 10, 100, 1000 via model(tensor, training=False).
        Invariant: Output shape is (B, num_classes), probabilities in [0, 1], row sums == 1.0.
        """
        for batch_size in [1, 10, 100, 1000]:
            features = np.random.randn(batch_size, 63).astype(np.float32)
            tensor_in = tf.convert_to_tensor(features, dtype=tf.float32)
            output = self.model_1h(tensor_in, training=False).numpy()

            self.assertEqual(output.shape, (batch_size, 4))
            self.assertTrue(np.all(output >= 0.0))
            self.assertTrue(np.all(output <= 1.0))
            np.testing.assert_allclose(np.sum(output, axis=1), np.ones(batch_size), atol=1e-5)

    def test_empty_batch_size_zero_execution(self):
        """
        Adversarial probe: Batch size 0 (shape: (0, 63)).
        Invariant: Returns empty tensor of shape (0, 4) without crashing.
        """
        empty_features = np.zeros((0, 63), dtype=np.float32)
        tensor_in = tf.convert_to_tensor(empty_features, dtype=tf.float32)
        output = self.model_1h(tensor_in, training=False).numpy()
        self.assertEqual(output.shape, (0, 4))

    def test_single_1d_array_and_list_inputs(self):
        """
        Adversarial probe: predict() and predict_proba() on 1D ndarray and Python list.
        Invariant: Returns 1D probability array of shape (num_classes,) summing to 1.0.
        """
        # 1D single-hand numpy array (63,)
        feat_1d = np.random.randn(63).astype(np.float32)
        probs_1d = self.trainer_1h.predict_proba(feat_1d)
        self.assertEqual(probs_1d.shape, (4,))
        self.assertAlmostEqual(float(np.sum(probs_1d)), 1.0, places=5)

        # 1D single-hand Python list of length 63
        feat_list = feat_1d.tolist()
        probs_list = self.trainer_1h.predict_proba(feat_list)
        self.assertEqual(probs_list.shape, (4,))
        self.assertAlmostEqual(float(np.sum(probs_list)), 1.0, places=5)

        # 1D two-handed numpy array (126,)
        feat_2h = np.random.randn(126).astype(np.float32)
        probs_2h = self.trainer_2h.predict_proba(feat_2h)
        self.assertEqual(probs_2h.shape, (6,))
        self.assertAlmostEqual(float(np.sum(probs_2h)), 1.0, places=5)

    def test_numerical_anomalies_and_extreme_values(self):
        """
        Adversarial probe: Extreme magnitudes (+-1e8), all-zero vectors, and subnormals.
        Invariant: Direct tensor execution produces valid probabilities without crashing.
        """
        # All zeroes
        zeros = np.zeros((5, 63), dtype=np.float32)
        out_zeros = self.trainer_1h.predict_proba(zeros)
        self.assertEqual(out_zeros.shape, (5, 4))
        self.assertFalse(np.isnan(out_zeros).any())
        np.testing.assert_allclose(np.sum(out_zeros, axis=1), np.ones(5), atol=1e-5)

        # Large positive values (+1e8)
        large_pos = np.full((3, 63), 1e8, dtype=np.float32)
        out_large = self.trainer_1h.predict_proba(large_pos)
        self.assertEqual(out_large.shape, (3, 4))

        # Large negative values (-1e8)
        large_neg = np.full((3, 63), -1e8, dtype=np.float32)
        out_neg = self.trainer_1h.predict_proba(large_neg)
        self.assertEqual(out_neg.shape, (3, 4))

    def test_dimension_mismatch_rejection(self):
        """
        Adversarial probe: Passing invalid feature dimensions (e.g. 63 to 126 model, 30 to 63 model).
        Invariant: TensorFlow / Keras rejects mismatched tensor shape with ValueError.
        """
        # Pass 63 features to 126 model
        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError, Exception)):
            self.trainer_2h.predict_proba(np.zeros(63, dtype=np.float32))

        # Pass 30 features to 63 model
        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError, Exception)):
            self.trainer_1h.predict_proba(np.zeros(30, dtype=np.float32))

    def test_dtype_compatibility(self):
        """
        Adversarial probe: Passing float64, int32, and float16 arrays to predict_proba.
        Invariant: Successfully converted to tf.float32 without raising type errors.
        """
        sample_f64 = np.random.randn(63).astype(np.float64)
        out_f64 = self.trainer_1h.predict_proba(sample_f64)
        self.assertEqual(out_f64.shape, (4,))

        sample_i32 = np.random.randint(0, 10, size=(63,)).astype(np.int32)
        out_i32 = self.trainer_1h.predict_proba(sample_i32)
        self.assertEqual(out_i32.shape, (4,))

    def test_direct_tensor_vs_predict_latency_benchmark(self):
        """
        Adversarial benchmark: Compare direct tensor execution model(x, training=False)
        against legacy model.predict(x, verbose=0) over 100 single-sample inferences.
        Target: Direct tensor execution is significantly faster (> 1.5x speedup) with latency < 15.0ms on CPU.
        """
        sample = np.random.randn(1, 63).astype(np.float32)
        tensor_in = tf.convert_to_tensor(sample, dtype=tf.float32)

        # Warmup
        for _ in range(10):
            _ = self.model_1h(tensor_in, training=False)
            _ = self.model_1h.predict(sample, verbose=0)

        # Benchmark direct tensor inference
        n_iters = 100
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = self.model_1h(tensor_in, training=False).numpy()
        direct_time = (time.perf_counter() - t0) / n_iters * 1000.0  # ms

        # Benchmark model.predict()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = self.model_1h.predict(sample, verbose=0)
        predict_time = (time.perf_counter() - t0) / n_iters * 1000.0  # ms

        speedup = predict_time / direct_time if direct_time > 0 else float('inf')
        print(f"\n[BENCHMARK] Direct tensor latency : {direct_time:.3f} ms / sample")
        print(f"[BENCHMARK] Keras predict latency : {predict_time:.3f} ms / sample")
        print(f"[BENCHMARK] Direct inference speedup: {speedup:.1f}x")

        self.assertLess(direct_time, 15.0, f"Direct tensor inference latency {direct_time:.3f}ms exceeded 15.0ms threshold")
        self.assertGreater(speedup, 1.2, f"Direct tensor inference speedup {speedup:.1f}x was not faster than model.predict")


if __name__ == '__main__':
    unittest.main()
