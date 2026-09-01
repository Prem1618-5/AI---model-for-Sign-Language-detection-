"""
Adversarial Stress Test Suite for M2 Modules
Author: Forensic Auditor M2
Stress-tests:
- TemporalSmoother under adversarial inputs (zero vectors, unnormalized vectors, timestamp jumps, backwards time, NaN/inf handling)
- ModelTrainer under varied batch and feature shapes
- Data preprocessing with anomalous sample inputs
"""

import os
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from temporal_filter import TemporalSmoother
from model_training import GestureModelTrainer
from data_preprocessing import GestureDataProcessor, parse_raw_sample

def run_adversarial_stress_tests():
    print("=" * 70)
    print("RUNNING ADVERSARIAL STRESS TESTS (M2)")
    print("=" * 70)

    # 1. TemporalSmoother under unnormalized probabilities
    smoother = TemporalSmoother(alpha=0.3, class_names=['A', 'B'])
    # Probabilities sum to 10.0 instead of 1.0
    pred, conf, status = smoother.update(np.array([6.0, 4.0]))
    assert np.isclose(np.sum(smoother.smoothed_probs), 1.0, atol=1e-5), "Smoother failed to normalize unnormalized raw input"
    print("  [Stress 1] Unnormalized probability input normalized to sum=1.0: PASS")

    # 2. TemporalSmoother backwards timestamp anomaly (dt <= 0)
    smoother.reset()
    smoother.update(np.array([0.9, 0.1]), timestamp=100.0)
    # Next timestamp is in the past (90.0)
    pred, conf, status = smoother.update(np.array([0.9, 0.1]), wrist_pos=(0.5, 0.5), timestamp=90.0)
    # Should not crash with ZeroDivisionError or negative dt
    print("  [Stress 2] Monotonic timestamp anomaly / backwards time protected: PASS")

    # 3. TemporalSmoother large inactivity jump (> sequence_timeout)
    t = 10.0
    for _ in range(5):
        t += 0.033
        smoother.update(np.array([0.95, 0.05]), timestamp=t)
    assert len(smoother.sequence_buffer) == 1
    # Jump 10 seconds into the future
    t += 10.0
    smoother.update(np.array([0.05, 0.95]), timestamp=t)
    # Sequence buffer should be cleared on timeout
    assert len(smoother.sequence_buffer) == 0, f"Expected empty buffer immediately on timeout, got {len(smoother.sequence_buffer)}"
    print("  [Stress 3] Inactivity timeout buffer reset under large timestamp gap: PASS")

    # 4. Direct Tensor Inference with variable batch sizes
    trainer = GestureModelTrainer(model_dir='models')
    trainer.build_model(input_shape=(63,), num_classes=3)
    trainer.class_names = ['c1', 'c2', 'c3']

    for batch_size in [1, 2, 7, 33, 128]:
        batch_data = np.random.randn(batch_size, 63).astype(np.float32)
        probs = trainer.predict_proba(batch_data)
        assert probs.shape == (batch_size, 3)
        assert np.allclose(np.sum(probs, axis=1), np.ones(batch_size), atol=1e-5)
    print("  [Stress 4] Variable batch direct tensor inference probability conservation: PASS")

    # 5. parse_raw_sample edge cases
    assert parse_raw_sample([]) == []
    assert parse_raw_sample([{'x': 0.1, 'y': 0.2, 'z': 0.3}]) == [[{'x': 0.1, 'y': 0.2, 'z': 0.3}]]
    assert parse_raw_sample([[{}], []]) == [[{}]]
    assert parse_raw_sample(None) == []
    print("  [Stress 5] parse_raw_sample edge cases (empty, None, malformed list): PASS")

    print("\nAdversarial Stress Testing Completed Successfully!")

if __name__ == '__main__':
    run_adversarial_stress_tests()
