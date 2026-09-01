"""
Real-time Recognition Module for Sign Language Detection ML Project

This module coordinates webcam capture, landmark normalization, ML inference,
temporal stability filtering, and delegates visual rendering to SignLanguageHUD.
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from datetime import datetime

from data_preprocessing import GestureDataProcessor
from model_training import GestureModelTrainer
from temporal_filter import TemporalSmoother
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


class RealtimeGestureRecognizer:
    """
    Coordinates real-time gesture recognition using MediaPipe,
    TensorFlow direct inference, temporal filtering, and SignLanguageHUD.
    """

    def __init__(self,
                 model_path=None,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 recognition_threshold=0.7,
                 smoothing_window=10):
        """
        Initialize the RealtimeGestureRecognizer.

        Args:
            model_path (str): Path to the trained model
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
            recognition_threshold (float): Minimum confidence threshold for gesture recognition
            smoothing_window (int): Window size for temporal smoothing
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Model setup
        self.trainer = GestureModelTrainer()
        if model_path:
            self.model = self.trainer.load_model(model_path)
        else:
            self.model = self.trainer.load_model()

        self.class_names = self.trainer.class_names

        # Recognition parameters
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.recognition_threshold = recognition_threshold

        # Temporal smoother (Softmax EMA + Hysteresis + Kinematic Velocity Gate)
        self.smoother = TemporalSmoother(
            alpha=0.25,
            threshold_high=recognition_threshold,
            threshold_low=0.45,
            debounce_frames=4,
            velocity_threshold=0.08,
            sequence_timeout=2.0,
            class_names=self.class_names
        )

        # UI Overlay Renderer
        self._hud = SignLanguageHUD(class_names=self.class_names)

        # Legacy smoothing buffer and sequence compatibility
        self.smoothing_window = smoothing_window
        self.history_buffer = deque(maxlen=smoothing_window)
        self.sequence_buffer = []
        self.sequence_timeout = 2.0
        self.last_gesture_time = 0.0

        # Processor for landmark normalization
        self.processor = GestureDataProcessor()

        # Flag to track two-handed gestures
        self.is_two_handed_model = self.trainer.is_two_handed

        # FPS tracking
        self._fps_buffer = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Frame counter for animations
        self._frame_count = 0

    @property
    def hud(self) -> SignLanguageHUD:
        """Lazily initialize HUD renderer if needed."""
        if not hasattr(self, '_hud') or self._hud is None:
            classes = getattr(self, 'class_names', [])
            self._hud = SignLanguageHUD(class_names=classes)
        return self._hud

    @hud.setter
    def hud(self, value: SignLanguageHUD):
        self._hud = value

    # ── Landmark preprocessing ──────────────────────────────────────────

    def preprocess_landmarks(self, landmarks):
        """
        Preprocess hand landmarks for model input.

        Args:
            landmarks (list): MediaPipe hand landmarks

        Returns:
            numpy.ndarray: Processed features for model input
        """
        landmark_list = []
        for landmark in landmarks.landmark:
            landmark_list.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': 1.0
            })

        normalized = self.processor.normalize_landmarks(landmark_list)
        features = self.processor.flatten_landmarks(normalized)
        return features

    def preprocess_two_hands(self, left_hand_landmarks, right_hand_landmarks):
        """
        Preprocess landmarks from both hands together.

        Args:
            left_hand_landmarks: MediaPipe landmarks for left hand (or None)
            right_hand_landmarks: MediaPipe landmarks for right hand (or None)

        Returns:
            numpy.ndarray: Combined processed features for model input (126-dim)
        """
        left_features = self.preprocess_landmarks(left_hand_landmarks) if left_hand_landmarks else np.zeros(63)
        right_features = self.preprocess_landmarks(right_hand_landmarks) if right_hand_landmarks else np.zeros(63)
        return np.concatenate([left_features, right_features])

    def preprocess_single_hand_for_two_handed_model(self, hand_landmarks):
        """
        Preprocess a single hand for a two-handed model by zero-padding.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            numpy.ndarray: 126-dim feature vector (63 real + 63 zeros)
        """
        features = self.preprocess_landmarks(hand_landmarks)
        return np.concatenate([features, np.zeros(63)])

    # ── Smoothing & sequence ────────────────────────────────────────────

    def get_smoothed_prediction(self):
        """
        Get smoothed prediction based on recent history.

        Returns:
            tuple: (gesture_name, confidence) or (None, 0) if no clear prediction
        """
        if not hasattr(self, 'history_buffer') or not self.history_buffer:
            return None, 0.0

        gesture_counts = {}
        gesture_confidences = {}

        for gesture, confidence in self.history_buffer:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                gesture_confidences[gesture] = 0.0
            gesture_counts[gesture] += 1
            gesture_confidences[gesture] += confidence

        max_count = 0
        max_gesture = None
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                max_gesture = gesture

        if max_count / len(self.history_buffer) >= 0.6:
            avg_confidence = gesture_confidences[max_gesture] / max_count
            return max_gesture, avg_confidence

        return None, 0.0

    def update_sequence(self, gesture, confidence):
        """
        Update gesture sequence with new detection.

        Args:
            gesture (str): Recognized gesture
            confidence (float): Recognition confidence
        """
        current_time = time.time()

        if current_time - self.last_gesture_time > self.sequence_timeout:
            self.sequence_buffer = []

        self.last_gesture_time = current_time

        if not self.sequence_buffer or self.sequence_buffer[-1][0] != gesture:
            self.sequence_buffer.append((gesture, confidence))
            if len(self.sequence_buffer) > 10:
                self.sequence_buffer.pop(0)

    def get_sequence_text(self):
        """
        Get current gesture sequence as text.

        Returns:
            str: Arrow-separated gesture sequence
        """
        if not hasattr(self, 'sequence_buffer') or not self.sequence_buffer:
            return ""
        return '  >  '.join([item[0].upper() for item in self.sequence_buffer])

    # ── Delegated UI Drawing Helpers (for backwards compatibility) ───────

    def _overlay_rect(self, image, x, y, w, h, colour=COL_BG, alpha=0.80):
        """Delegate ROI alpha blending to HUD renderer."""
        self.hud._overlay_rect(image, x, y, w, h, colour, alpha)

    def _draw_rounded_rect(self, image, x, y, w, h, colour, thickness=1, radius=8):
        """Delegate rounded rectangle drawing to HUD renderer."""
        self.hud._draw_rounded_rect(image, x, y, w, h, colour, thickness, radius)

    def draw_confidence_bar(self, image, x, y, w, h, confidence):
        """Delegate confidence bar drawing to HUD renderer."""
        self.hud.draw_confidence_bar(image, x, y, w, h, confidence)

    def draw_hand_skeleton(self, image, hand_landmarks):
        """Delegate hand skeleton drawing to HUD renderer."""
        self.hud.draw_hand_skeleton(image, hand_landmarks)

    def draw_gesture_legend(self, image, x=None, y=None):
        """Delegate gesture legend sidebar to HUD renderer."""
        if x is None:
            w = image.shape[1]
            x = w - 175
        if y is None:
            y = 55
        self.hud.draw_gesture_legend(image, x, y)

    def draw_top_bar(self, image, fps):
        """Delegate top bar drawing to HUD renderer."""
        self.hud.draw_top_bar(image, fps)

    def draw_detection_panel(self, image, prediction_text, confidence, status):
        """Delegate detection panel drawing to HUD renderer."""
        self.hud.draw_detection_panel(image, prediction_text, confidence, status)

    def draw_sequence_panel(self, image, sequence_text):
        """Delegate sequence panel drawing to HUD renderer."""
        self.hud.draw_sequence_panel(image, sequence_text)

    def draw_controls_bar(self, image):
        """Delegate controls footer drawing to HUD renderer."""
        self.hud.draw_controls_bar(image)

    # ── Main recognition loop ──────────────────────────────────────────

    def run(self, camera_id=0, flip_image=True):
        """
        Run real-time gesture recognition with webcam feed.

        Args:
            camera_id (int): Camera device ID
            flip_image (bool): Whether to flip the camera image horizontally
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide a valid model path.")
        if not self.class_names:
            raise ValueError("Class names not available. Check model metadata.")

        print("Starting real-time gesture recognition...")
        print(f"Loaded {len(self.class_names)} gestures: {', '.join(self.class_names)}")
        print(f"Two-handed model: {self.is_two_handed_model}")
        print(f"Camera ID: {camera_id}, Flip image: {flip_image}")
        print("Press 'q' to quit, 'c' to clear sequence, 's' to screenshot")

        # Initialize webcam
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera with ID {camera_id}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("Camera opened successfully.")
        except Exception as e:
            print(f"Error opening camera: {e}")
            return

        try:
            cv2.namedWindow('Sign Language AI', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Sign Language AI', 1280, 720)

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    time.sleep(0.1)
                    continue

                self._frame_count += 1

                # FPS calculation
                now = time.time()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0:
                    self._fps_buffer.append(1.0 / dt)
                fps = float(np.mean(self._fps_buffer)) if self._fps_buffer else 0.0

                # Flip horizontal if configured
                if flip_image:
                    image = cv2.flip(image, 1)

                h_img, w_img, _ = image.shape

                # Process with MediaPipe
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                image.flags.writeable = True

                # ── Prediction & Landmark Processing ────────────────
                confidence = 0.0
                status = "SCANNING"
                smooth_gesture = "None"
                hands_info = []

                if results.multi_hand_landmarks:
                    left_hand_landmarks = None
                    right_hand_landmarks = None

                    # Extract primary hand wrist position for kinematic velocity gating
                    primary_wrist = results.multi_hand_landmarks[0].landmark[0]
                    wrist_pos = (primary_wrist.x, primary_wrist.y)

                    # Classify handedness and collect hands_info
                    if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            hand_label = handedness.classification[0].label
                            score = float(handedness.classification[0].score)
                            if hand_label == "Left":
                                left_hand_landmarks = hand_landmarks
                            else:
                                right_hand_landmarks = hand_landmarks

                            xs = [lm.x for lm in hand_landmarks.landmark]
                            ys = [lm.y for lm in hand_landmarks.landmark]
                            pad = 14
                            bbox = (
                                max(0, int(min(xs) * w_img) - pad),
                                max(0, int(min(ys) * h_img) - pad),
                                min(w_img, int(max(xs) * w_img) + pad),
                                min(h_img, int(max(ys) * h_img) + pad)
                            )
                            hands_info.append({
                                'landmarks': hand_landmarks,
                                'handedness': hand_label,
                                'score': score,
                                'bbox': bbox
                            })
                    else:
                        for hand_landmarks in results.multi_hand_landmarks:
                            xs = [lm.x for lm in hand_landmarks.landmark]
                            ys = [lm.y for lm in hand_landmarks.landmark]
                            pad = 14
                            bbox = (
                                max(0, int(min(xs) * w_img) - pad),
                                max(0, int(min(ys) * h_img) - pad),
                                min(w_img, int(max(xs) * w_img) + pad),
                                min(h_img, int(max(ys) * h_img) + pad)
                            )
                            hands_info.append({
                                'landmarks': hand_landmarks,
                                'handedness': None,
                                'score': None,
                                'bbox': bbox
                            })

                    # Build feature vector for ML inference
                    features = None
                    if self.is_two_handed_model:
                        if left_hand_landmarks and right_hand_landmarks:
                            features = self.preprocess_two_hands(left_hand_landmarks, right_hand_landmarks)
                        elif len(results.multi_hand_landmarks) == 2:
                            features = self.preprocess_two_hands(
                                results.multi_hand_landmarks[0],
                                results.multi_hand_landmarks[1]
                            )
                        elif len(results.multi_hand_landmarks) == 1:
                            features = self.preprocess_single_hand_for_two_handed_model(
                                results.multi_hand_landmarks[0]
                            )
                    else:
                        features = self.preprocess_landmarks(results.multi_hand_landmarks[0])

                    if features is not None:
                        # Direct tensor inference returning full softmax distribution
                        raw_probs = self.trainer.predict_proba(features)

                        # Update temporal filter with Softmax EMA, hysteresis, velocity gate
                        smooth_gesture, smooth_conf, status = self.smoother.update(
                            raw_probs, wrist_pos=wrist_pos, timestamp=now
                        )
                        confidence = smooth_conf

                        # Update legacy buffers for backward compatibility
                        self.sequence_buffer = list(self.smoother.sequence_buffer)
                        self.history_buffer.append((smooth_gesture, smooth_conf))
                else:
                    self.smoother.reset()

                # Sequence timeout countdown calculation
                seq_progress = 0.0
                if self.smoother.last_gesture_time > 0 and self.smoother.sequence_buffer:
                    elapsed = now - self.smoother.last_gesture_time
                    seq_progress = max(0.0, 1.0 - (elapsed / self.smoother.sequence_timeout))

                # ── Construct HUDState & Render Overlay ──────────────
                state = HUDState(
                    fps=fps,
                    detected=(status == "DETECTED"),
                    status_text=status,
                    gesture=smooth_gesture if status == "DETECTED" else ("Analysing..." if status == "UNCERTAIN" else "None"),
                    confidence=confidence,
                    sequence=self.smoother.get_sequence(),
                    hands_info=hands_info,
                    classes=self.class_names,
                    sequence_progress=seq_progress
                )

                # Composite HUD overlay via SignLanguageHUD
                image = self.hud.render(image, state)

                # Display frame
                cv2.imshow('Sign Language AI', image)

                # Keyboard handling
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed.")
                    break
                elif key == ord('c'):
                    self.smoother.reset()
                    self.smoother.sequence_buffer.clear()
                    self.sequence_buffer = []
                    self.history_buffer.clear()
                    print("Sequence and history cleared.")
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{ts}.png"
                    cv2.imwrite(screenshot_path, image)
                    print(f"Screenshot saved: {screenshot_path}")

        except Exception as e:
            print(f"Error in recognition loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            print("Recognition stopped.")


if __name__ == "__main__":
    print("Starting standalone real-time recognition...")
    models_dir = '../models'
    model_metadata_path = os.path.join(models_dir, 'model_metadata.json')

    if os.path.exists(model_metadata_path):
        print("Found model metadata, initializing recognizer...")
        recognizer = RealtimeGestureRecognizer()
        try:
            print("Starting recognition...")
            recognizer.run()
        except Exception as e:
            print(f"Error during recognition: {e}")
    else:
        print("No trained model found. Please train a model first using model_training.py")
        print("You can use data_collection.py to collect gesture data, then process and train a model.")