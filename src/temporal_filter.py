"""
Temporal Prediction & Smoothing Filter Module

Provides continuous Softmax Exponential Moving Average (EMA) probability smoothing,
dual-threshold hysteresis debouncing, kinematic wrist velocity gating, and
sequence buffer management with inactivity timeouts.
"""

import time
import numpy as np


class TemporalSmoother:
    """
    Temporal prediction smoother incorporating:
    1. Continuous Softmax Exponential Moving Average (EMA)
    2. Kinematic wrist velocity gating
    3. Dual-threshold hysteresis state machine
    4. Sequence tracking with deduplication and inactivity timeout
    """

    def __init__(self,
                 alpha: float = 0.25,
                 threshold_high: float = 0.80,
                 threshold_low: float = 0.45,
                 debounce_frames: int = 4,
                 velocity_threshold: float = 0.08,
                 sequence_timeout: float = 2.0,
                 class_names: list[str] | None = None):
        """
        Initialize TemporalSmoother.

        Args:
            alpha (float): EMA smoothing factor (0.0 to 1.0). Default 0.25.
            threshold_high (float): High confidence threshold to enter DETECTED state. Default 0.80.
            threshold_low (float): Low confidence threshold to remain in DETECTED state. Default 0.45.
            debounce_frames (int): Number of consecutive frames >= threshold_high required. Default 4.
            velocity_threshold (float): Maximum allowed wrist displacement/velocity before gating. Default 0.08.
            sequence_timeout (float): Inactivity timeout in seconds to reset sequence. Default 2.0.
            class_names (list): List of gesture class names.
        """
        self.alpha = float(alpha)
        self.threshold_high = float(threshold_high)
        self.threshold_low = float(threshold_low)
        self.debounce_frames = int(debounce_frames)
        self.velocity_threshold = float(velocity_threshold)
        self.sequence_timeout = float(sequence_timeout)
        self.class_names = list(class_names) if class_names else []

        self.smoothed_probs: np.ndarray | None = None
        self.current_state: str = "SCANNING"  # "SCANNING", "DETECTED", "UNCERTAIN"
        self.active_class: str = "None"
        self.active_confidence: float = 0.0
        self.consecutive_high_frames: int = 0
        self.last_wrist_pos: tuple[float, float] | None = None
        self.last_update_time: float = 0.0

        self.sequence_buffer: list[tuple[str, float]] = []
        self.last_gesture_time: float = 0.0

    def reset(self):
        """Reset temporal smoothing state (called when hand tracking is lost)."""
        self.smoothed_probs = None
        self.current_state = "SCANNING"
        self.active_class = "None"
        self.active_confidence = 0.0
        self.consecutive_high_frames = 0
        self.last_wrist_pos = None

    def update(self,
               raw_probs: np.ndarray | list[float],
               wrist_pos: tuple[float, float] | None = None,
               timestamp: float | None = None) -> tuple[str, float, str]:
        """
        Update smoother with raw softmax probabilities and optional wrist coordinates.

        Args:
            raw_probs (numpy.ndarray or list): Raw softmax probability vector.
            wrist_pos (tuple[float, float], optional): Normalized wrist (x, y) coordinates.
            timestamp (float, optional): Timestamp in seconds (defaults to time.time()).

        Returns:
            tuple: (predicted_class, smoothed_confidence, status_text)
                   where status_text is one of "DETECTED", "UNCERTAIN", "SCANNING".
        """
        current_time = timestamp if timestamp is not None else time.time()
        dt = max(current_time - self.last_update_time, 1e-4) if self.last_update_time > 0 else 0.033

        # Inactivity timeout detection: clear sequence if idle > sequence_timeout
        if self.last_update_time > 0 and (current_time - self.last_update_time > self.sequence_timeout):
            self.sequence_buffer.clear()
            self.smoothed_probs = None
            self.current_state = "SCANNING"
            self.consecutive_high_frames = 0

        self.last_update_time = current_time

        # 1. Softmax EMA Probability Smoothing: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
        raw_arr = np.asarray(raw_probs, dtype=np.float32)
        if self.smoothed_probs is None or len(self.smoothed_probs) != len(raw_arr):
            self.smoothed_probs = raw_arr.copy()
        else:
            self.smoothed_probs = self.alpha * raw_arr + (1.0 - self.alpha) * self.smoothed_probs

        # Ensure probability distribution conserves sum == 1.0
        prob_sum = float(np.sum(self.smoothed_probs))
        if prob_sum > 0:
            self.smoothed_probs = self.smoothed_probs / prob_sum

        top_idx = int(np.argmax(self.smoothed_probs))
        top_conf = float(self.smoothed_probs[top_idx])
        top_class = self.class_names[top_idx] if (self.class_names and top_idx < len(self.class_names)) else str(top_idx)

        # 2. Kinematic Wrist Velocity Gating
        is_moving_fast = False
        if wrist_pos is not None:
            if self.last_wrist_pos is not None:
                dx = wrist_pos[0] - self.last_wrist_pos[0]
                dy = wrist_pos[1] - self.last_wrist_pos[1]
                displacement = np.sqrt(dx * dx + dy * dy)
                velocity = displacement / dt if dt > 0 else displacement
                # Suppress if inter-frame displacement or velocity exceeds threshold
                if displacement > self.velocity_threshold or velocity > self.velocity_threshold:
                    is_moving_fast = True
            self.last_wrist_pos = wrist_pos

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
                # Drop below low threshold -> return to SCANNING
                self.current_state = "SCANNING"
                self.active_class = "None"
                self.consecutive_high_frames = 0
                return ("None", top_conf, "SCANNING")
        else:
            # In SCANNING / UNCERTAIN state: requires sustained confidence >= T_high for debounce_frames
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
        """Append to gesture sequence with deduplication and inactivity timeout."""
        if self.last_gesture_time > 0 and (current_time - self.last_gesture_time > self.sequence_timeout):
            self.sequence_buffer.clear()

        self.last_gesture_time = current_time

        # Deduplicate consecutive identical gestures
        if not self.sequence_buffer or self.sequence_buffer[-1][0] != gesture:
            self.sequence_buffer.append((gesture, confidence))
            if len(self.sequence_buffer) > 10:
                self.sequence_buffer.pop(0)

    def get_sequence(self) -> list[str]:
        """Get sequence of recognized gesture names."""
        return [item[0].upper() for item in self.sequence_buffer]

    def get_sequence_text(self, delimiter: str = "  >  ") -> str:
        """Get formatted gesture sequence string."""
        return delimiter.join(self.get_sequence())
