"""
Temporal Prediction & Smoothing Filter Test Suite
Tests Softmax Exponential Moving Average (EMA) probability smoothing,
dual-threshold hysteresis debouncing, kinematic wrist velocity gating,
and sequence buffer timeout management.
"""

import os
import sys
import time
import unittest
import numpy as np

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from collections import deque
from realtime_recognition import RealtimeGestureRecognizer
from temporal_filter import TemporalSmoother


class TemporalFilterReference:
    """
    Reference mathematical implementation of the Temporal Prediction Filter
    specified in PROJECT.md and Milestone 2 contracts.
    """

    def __init__(self,
                 alpha: float = 0.25,
                 threshold_high: float = 0.80,
                 threshold_low: float = 0.45,
                 debounce_frames: int = 4,
                 velocity_threshold: float = 0.20,
                 sequence_timeout: float = 2.0,
                 class_names: list[str] | None = None):
        self.alpha = alpha
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.debounce_frames = debounce_frames
        self.velocity_threshold = velocity_threshold
        self.sequence_timeout = sequence_timeout
        self.class_names = class_names or []

        self.smoothed_probs = None
        self.current_state = "SCANNING"  # "SCANNING", "DETECTED", "UNCERTAIN"
        self.active_class = "None"
        self.active_confidence = 0.0
        self.consecutive_high_frames = 0
        self.last_wrist_pos = None
        self.last_update_time = time.time()

        self.sequence_buffer: list[tuple[str, float]] = []
        self.last_gesture_time = 0.0

    def reset(self):
        """Reset temporal state when tracking is lost."""
        self.smoothed_probs = None
        self.current_state = "SCANNING"
        self.active_class = "None"
        self.active_confidence = 0.0
        self.consecutive_high_frames = 0
        self.last_wrist_pos = None

    def update(self,
               raw_probs: np.ndarray,
               wrist_pos: tuple[float, float] | None = None,
               timestamp: float | None = None) -> tuple[str, float, str]:
        """
        Updates filter with frame raw probabilities and optional wrist coordinates.
        Returns: (predicted_class, smoothed_confidence, status_text)
        """
        current_time = timestamp if timestamp is not None else time.time()
        dt = max(current_time - self.last_update_time, 1e-4)

        # Inactivity timeout detection
        if current_time - self.last_update_time > self.sequence_timeout and self.last_update_time > 0:
            self.sequence_buffer.clear()
            self.smoothed_probs = None
            self.current_state = "SCANNING"
            self.consecutive_high_frames = 0

        self.last_update_time = current_time

        # 1. Softmax EMA Probability Smoothing: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
        if self.smoothed_probs is None:
            self.smoothed_probs = np.array(raw_probs, dtype=np.float32)
        else:
            self.smoothed_probs = (self.alpha * np.array(raw_probs, dtype=np.float32) +
                                  (1.0 - self.alpha) * self.smoothed_probs)

        # Normalize to maintain probability sum == 1.0
        prob_sum = np.sum(self.smoothed_probs)
        if prob_sum > 0:
            self.smoothed_probs = self.smoothed_probs / prob_sum

        top_idx = int(np.argmax(self.smoothed_probs))
        top_conf = float(self.smoothed_probs[top_idx])
        top_class = self.class_names[top_idx] if top_idx < len(self.class_names) else str(top_idx)

        # 2. Kinematic Wrist Velocity Gating
        is_moving_fast = False
        if wrist_pos is not None:
            if self.last_wrist_pos is not None:
                dx = wrist_pos[0] - self.last_wrist_pos[0]
                dy = wrist_pos[1] - self.last_wrist_pos[1]
                velocity = np.sqrt(dx * dx + dy * dy) / dt
                if velocity > self.velocity_threshold:
                    is_moving_fast = True
            self.last_wrist_pos = wrist_pos

        # If hand is moving too fast during transition, suppress detection
        if is_moving_fast:
            self.consecutive_high_frames = 0
            if self.current_state == "DETECTED":
                self.current_state = "UNCERTAIN"
            return ("Analysing...", top_conf, "UNCERTAIN")

        # 3. Dual-Threshold Hysteresis State Machine
        if self.current_state == "DETECTED":
            # In DETECTED state: keep holding as long as confidence stays above T_low
            if top_conf >= self.threshold_low:
                self.active_class = top_class
                self.active_confidence = top_conf
                self._update_sequence(top_class, top_conf, current_time)
                return (self.active_class, self.active_confidence, "DETECTED")
            else:
                # Drop below low threshold -> release detection
                self.current_state = "SCANNING"
                self.active_class = "None"
                self.consecutive_high_frames = 0
                return ("None", top_conf, "SCANNING")
        else:
            # In SCANNING / UNCERTAIN state: requires sustained confidence >= T_high
            if top_conf >= self.threshold_high:
                self.consecutive_high_frames += 1
                if self.consecutive_high_frames >= self.debounce_frames:
                    self.current_state = "DETECTED"
                    self.active_class = top_class
                    self.active_confidence = top_conf
                    self._update_sequence(top_class, top_conf, current_time)
                    return (self.active_class, self.active_confidence, "DETECTED")
                else:
                    return ("Analysing...", top_conf, "UNCERTAIN")
            else:
                self.consecutive_high_frames = 0
                self.current_state = "SCANNING"
                return ("None", top_conf, "SCANNING")

    def _update_sequence(self, gesture: str, confidence: float, current_time: float):
        """Append to sequence with duplicate suppression and timeout."""
        if current_time - self.last_gesture_time > self.sequence_timeout and self.last_gesture_time > 0:
            self.sequence_buffer.clear()

        self.last_gesture_time = current_time

        if not self.sequence_buffer or self.sequence_buffer[-1][0] != gesture:
            self.sequence_buffer.append((gesture, confidence))
            if len(self.sequence_buffer) > 10:
                self.sequence_buffer.pop(0)


class TestTemporalFiltering(unittest.TestCase):
    """Tier 2: Algorithmic unit tests for EMA smoothing, dual hysteresis, and velocity gating."""

    def setUp(self):
        self.classes = ['hello', 'thank_you', 'yes', 'no']
        self.smoother = TemporalSmoother(
            alpha=0.25,
            threshold_high=0.80,
            threshold_low=0.45,
            debounce_frames=4,
            velocity_threshold=0.20,
            sequence_timeout=2.0,
            class_names=self.classes
        )

    def test_ema_smoothing_step_response(self):
        """Invariant: Sudden probability jump from 0.0 to 1.0 rises monotonically towards 1.0."""
        # Initial state: 100% yes
        raw_initial = np.array([0.0, 0.0, 1.0, 0.0])
        self.smoother.update(raw_initial)

        # Step jump: 100% hello
        raw_step = np.array([1.0, 0.0, 0.0, 0.0])
        prev_prob = 0.0

        for _ in range(10):
            self.smoother.update(raw_step)
            curr_prob = self.smoother.smoothed_probs[0]
            self.assertGreater(curr_prob, prev_prob, "EMA probability must rise monotonically on step input")
            self.assertAlmostEqual(float(np.sum(self.smoother.smoothed_probs)), 1.0, places=5)
            prev_prob = curr_prob

        # After 10 updates with alpha=0.25, should be close to 1.0
        self.assertGreater(self.smoother.smoothed_probs[0], 0.90)

    def test_ema_noise_filtering(self):
        """Invariant: Single-frame noise spike is dampened below high detection threshold."""
        # Baseline: steady 'yes' (index 2)
        for _ in range(5):
            self.smoother.update(np.array([0.05, 0.05, 0.85, 0.05]))

        # Single frame glitch/spike for 'hello' (index 0)
        self.smoother.update(np.array([0.95, 0.01, 0.02, 0.02]))

        # Smoothed probability for 'hello' must NOT immediately jump to >= 0.80
        hello_smoothed = self.smoother.smoothed_probs[0]
        self.assertLess(hello_smoothed, 0.40, "Single glitch frame must be dampened by EMA")

    def test_dual_threshold_hysteresis_debouncing(self):
        """Invariant: State requires T_high=0.80 + debounce frames to enter DETECTED, remains until < T_low=0.45."""
        high_prob = np.array([0.90, 0.03, 0.03, 0.04])

        # Frames 1 to 3: above 0.80, but fewer than debounce_frames (4) -> UNCERTAIN
        for i in range(1, 4):
            pred, conf, status = self.smoother.update(high_prob)
            self.assertEqual(status, "UNCERTAIN", f"Frame {i} should be UNCERTAIN before debounce threshold")

        # Frame 4: 4th consecutive frame >= 0.80 -> transitions to DETECTED
        pred, conf, status = self.smoother.update(high_prob)
        self.assertEqual(status, "DETECTED")
        self.assertEqual(pred, "hello")

        # Now confidence drops to 0.60 (between T_low=0.45 and T_high=0.80) -> MUST STAY DETECTED
        mid_prob = np.array([0.60, 0.15, 0.15, 0.10])
        pred, conf, status = self.smoother.update(mid_prob)
        self.assertEqual(status, "DETECTED", "Hysteresis must maintain DETECTED state when conf >= T_low (0.45)")

        # Confidence drops below T_low (e.g. 0.35) -> transitions to SCANNING / None
        low_prob = np.array([0.30, 0.25, 0.25, 0.20])
        # After smoothed prob drops below 0.45:
        for _ in range(5):
            pred, conf, status = self.smoother.update(low_prob)

        self.assertEqual(status, "SCANNING", "State must drop to SCANNING when conf < T_low (0.45)")

    def test_kinematic_velocity_gating(self):
        """Invariant: Rapid hand transit suppresses active detection even if classifier outputs high confidence."""
        # Steady hand at (0.5, 0.5) with high confidence -> reaches DETECTED
        t = 100.0
        for i in range(5):
            t += 0.033
            pred, conf, status = self.smoother.update(
                np.array([0.95, 0.02, 0.02, 0.01]),
                wrist_pos=(0.5, 0.5),
                timestamp=t
            )
        self.assertEqual(status, "DETECTED")

        # Sudden rapid motion (e.g. wrist moves from (0.5, 0.5) to (0.8, 0.8) in 33ms -> velocity ~ 12.8 >> 0.20)
        t += 0.033
        pred, conf, status = self.smoother.update(
            np.array([0.95, 0.02, 0.02, 0.01]),
            wrist_pos=(0.8, 0.8),
            timestamp=t
        )
        self.assertEqual(status, "UNCERTAIN", "Fast kinematic movement must gate prediction to UNCERTAIN")

    def test_sequence_buffer_deduplication_and_timeout(self):
        """Invariant: Sequence deduplicates repeated tokens and resets after sequence_timeout."""
        t = 1000.0
        # Produce 10 frames of confirmed 'hello'
        for _ in range(10):
            t += 0.05
            self.smoother.update(np.array([0.95, 0.01, 0.02, 0.02]), timestamp=t)

        self.assertEqual(len(self.smoother.sequence_buffer), 1)
        self.assertEqual(self.smoother.sequence_buffer[0][0], 'hello')

        # Switch to 'yes' (index 2) for 10 frames
        for _ in range(10):
            t += 0.05
            self.smoother.update(np.array([0.01, 0.01, 0.95, 0.03]), timestamp=t)

        self.assertEqual(len(self.smoother.sequence_buffer), 2)
        self.assertEqual(self.smoother.sequence_buffer[1][0], 'yes')

        # Inactivity for 3.0 seconds (> timeout 2.0s)
        t += 3.0
        # Trigger next gesture 'no' (index 3)
        for _ in range(10):
            t += 0.05
            self.smoother.update(np.array([0.01, 0.01, 0.02, 0.96]), timestamp=t)

        # Buffer must have cleared previous gestures after timeout
        self.assertEqual(len(self.smoother.sequence_buffer), 1)
        self.assertEqual(self.smoother.sequence_buffer[0][0], 'no')

    def test_temporal_smoother_reset_and_sequence_helpers(self):
        """Invariant: reset() restores initial state; get_sequence_text() produces formatted string."""
        # Drive into DETECTED state
        for _ in range(5):
            self.smoother.update(np.array([0.95, 0.02, 0.02, 0.01]))

        self.assertEqual(self.smoother.current_state, "DETECTED")
        self.assertEqual(self.smoother.get_sequence(), ["HELLO"])
        self.assertEqual(self.smoother.get_sequence_text(), "HELLO")

        # Call reset
        self.smoother.reset()
        self.assertIsNone(self.smoother.smoothed_probs)
        self.assertEqual(self.smoother.current_state, "SCANNING")
        self.assertEqual(self.smoother.consecutive_high_frames, 0)
        self.assertEqual(self.smoother.active_class, "None")


class TestRealtimeRecognizerBufferMethods(unittest.TestCase):
    """Tier 2: Unit tests for RealtimeGestureRecognizer buffer and sequence methods in isolation."""

    def test_history_buffer_smoothing_majority_vote(self):
        """Test RealtimeGestureRecognizer.get_smoothed_prediction majority logic."""
        # Create an uninitialized recognizer instance with mock attributes (no camera/model needed)
        rec = object.__new__(RealtimeGestureRecognizer)
        rec.history_buffer = deque(maxlen=10)

        # Empty buffer returns None, 0.0
        self.assertEqual(rec.get_smoothed_prediction(), (None, 0.0))

        # Fill with 7 'hello' (conf 0.9) and 3 'yes' (conf 0.8) -> majority >= 60%
        for _ in range(7):
            rec.history_buffer.append(('hello', 0.9))
        for _ in range(3):
            rec.history_buffer.append(('yes', 0.8))

        gesture, conf = rec.get_smoothed_prediction()
        self.assertEqual(gesture, 'hello')
        self.assertAlmostEqual(conf, 0.9, places=5)

        # Split 5 'hello' and 5 'yes' (50% < 60% threshold) -> returns None, 0.0
        rec.history_buffer.clear()
        for _ in range(5):
            rec.history_buffer.append(('hello', 0.9))
        for _ in range(5):
            rec.history_buffer.append(('yes', 0.9))

        gesture, conf = rec.get_smoothed_prediction()
        self.assertIsNone(gesture)
        self.assertEqual(conf, 0.0)

    def test_sequence_text_formatting(self):
        """Test RealtimeGestureRecognizer sequence text formatting."""
        rec = object.__new__(RealtimeGestureRecognizer)
        rec.sequence_buffer = [('hello', 0.95), ('thank_you', 0.88), ('yes', 0.92)]

        seq_text = rec.get_sequence_text()
        self.assertEqual(seq_text, "HELLO  >  THANK_YOU  >  YES")


if __name__ == '__main__':
    unittest.main()
