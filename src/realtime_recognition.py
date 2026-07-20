"""
Real-time Recognition Module for Sign Language Detection ML Project

This module handles real-time gesture recognition using a webcam,
allowing for interactive sign language detection with a premium HUD overlay.
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


# ── Colour palette ──────────────────────────────────────────────────────
COL_BG        = (15, 15, 30)         # dark navy background for panels
COL_CYAN      = (200, 220, 0)        # BGR cyan accent
COL_AMBER     = (0, 165, 255)        # BGR amber/orange
COL_GREEN     = (0, 220, 100)        # success green
COL_RED       = (60, 60, 220)        # muted red
COL_WHITE     = (240, 240, 240)
COL_GREY      = (140, 140, 140)
COL_DIM       = (80, 80, 90)
COL_BAR_BG    = (40, 40, 55)
COL_BAR_FILL  = (200, 220, 0)       # cyan fill
COL_JOINT     = (200, 220, 0)        # cyan joints
COL_CONN      = (160, 120, 0)        # blue-ish connections
COL_PULSE_A   = (200, 220, 0)        # pulse colour A (cyan)
COL_PULSE_B   = (0, 220, 200)        # pulse colour B (teal)


class RealtimeGestureRecognizer:
    """
    Handles real-time gesture recognition using webcam input
    with a premium HUD-style overlay UI.
    """

    def __init__(self,
                 model_path=None,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 recognition_threshold=0.7,
                 smoothing_window=10):
        """
        Initialize the RealtimeGestureRecognizer with specified parameters.

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

        # Smoothing and sequence detection
        self.smoothing_window = smoothing_window
        self.history_buffer = deque(maxlen=smoothing_window)
        self.sequence_buffer = []
        self.sequence_timeout = 2.0
        self.last_gesture_time = 0

        # Processor for landmark normalization
        self.processor = GestureDataProcessor()

        # Flag to track two-handed gestures
        self.is_two_handed_model = self.trainer.is_two_handed

        # FPS tracking
        self._fps_buffer = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Frame counter for animations
        self._frame_count = 0

    # ── Landmark preprocessing ──────────────────────────────────────────

    def preprocess_landmarks(self, landmarks, is_left_hand=None):
        """
        Preprocess hand landmarks for model input.

        Args:
            landmarks (list): MediaPipe hand landmarks
            is_left_hand (bool): Whether the hand is the left hand

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
        left_features = self.preprocess_landmarks(left_hand_landmarks, True) if left_hand_landmarks else np.zeros(63)
        right_features = self.preprocess_landmarks(right_hand_landmarks, False) if right_hand_landmarks else np.zeros(63)
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
        if not self.history_buffer:
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
        return '  >  '.join([item[0].upper() for item in self.sequence_buffer])

    # ── Drawing helpers ─────────────────────────────────────────────────

    def _overlay_rect(self, image, x, y, w, h, colour=COL_BG, alpha=0.80):
        """Draw a semi-transparent filled rectangle."""
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def _draw_rounded_rect(self, image, x, y, w, h, colour, thickness=1, radius=8):
        """Draw a rounded rectangle border."""
        # Top-left corner
        cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, colour, thickness)
        # Top-right corner
        cv2.ellipse(image, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, colour, thickness)
        # Bottom-right corner
        cv2.ellipse(image, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, colour, thickness)
        # Bottom-left corner
        cv2.ellipse(image, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, colour, thickness)
        # Lines
        cv2.line(image, (x + radius, y), (x + w - radius, y), colour, thickness)
        cv2.line(image, (x + radius, y + h), (x + w - radius, y + h), colour, thickness)
        cv2.line(image, (x, y + radius), (x, y + h - radius), colour, thickness)
        cv2.line(image, (x + w, y + radius), (x + w, y + h - radius), colour, thickness)

    def draw_confidence_bar(self, image, x, y, w, h, confidence):
        """Draw a styled confidence progress bar."""
        # Background
        cv2.rectangle(image, (x, y), (x + w, y + h), COL_BAR_BG, -1)
        # Fill
        fill_w = int(w * confidence)
        if confidence >= 0.7:
            fill_col = COL_GREEN
        elif confidence >= 0.4:
            fill_col = COL_AMBER
        else:
            fill_col = COL_RED
        if fill_w > 0:
            cv2.rectangle(image, (x, y), (x + fill_w, y + h), fill_col, -1)
        # Border
        cv2.rectangle(image, (x, y), (x + w, y + h), COL_DIM, 1)
        # Percentage text
        pct_text = f"{int(confidence * 100)}%"
        cv2.putText(image, pct_text, (x + w + 8, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    def draw_hand_skeleton(self, image, hand_landmarks):
        """Draw custom-coloured hand skeleton with styled joints and connections."""
        h, w, _ = image.shape
        connections = self.mp_hands.HAND_CONNECTIONS

        # Draw connections first
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            start_pt = (int(start.x * w), int(start.y * h))
            end_pt = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_pt, end_pt, COL_CONN, 2, cv2.LINE_AA)

        # Draw joints on top
        for i, landmark in enumerate(hand_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            # Fingertips get larger dots
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(image, (cx, cy), 6, COL_JOINT, -1, cv2.LINE_AA)
                cv2.circle(image, (cx, cy), 6, COL_WHITE, 1, cv2.LINE_AA)
            elif i == 0:
                # Wrist
                cv2.circle(image, (cx, cy), 5, COL_AMBER, -1, cv2.LINE_AA)
            else:
                cv2.circle(image, (cx, cy), 4, COL_JOINT, -1, cv2.LINE_AA)

    def draw_gesture_legend(self, image, x, y):
        """Draw the gesture legend sidebar listing loaded gesture classes."""
        pad = 10
        line_h = 24
        title_h = 30
        panel_h = title_h + len(self.class_names) * line_h + pad
        panel_w = 160

        self._overlay_rect(image, x, y, panel_w, panel_h, COL_BG, 0.75)
        self._draw_rounded_rect(image, x, y, panel_w, panel_h, COL_DIM, 1)

        # Title
        cv2.putText(image, "GESTURES", (x + pad, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1, cv2.LINE_AA)

        # List
        for i, name in enumerate(self.class_names):
            ty = y + title_h + i * line_h + 16
            cv2.circle(image, (x + pad + 4, ty - 4), 3, COL_GREEN, -1, cv2.LINE_AA)
            cv2.putText(image, name.capitalize(), (x + pad + 14, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)

    def draw_top_bar(self, image, fps):
        """Draw the top HUD bar with title and FPS."""
        h, w, _ = image.shape
        bar_h = 45
        self._overlay_rect(image, 0, 0, w, bar_h, COL_BG, 0.80)

        # Title
        cv2.putText(image, "Sign Language AI", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_CYAN, 2, cv2.LINE_AA)

        # FPS
        fps_text = f"FPS: {fps:.0f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(image, fps_text, (w - fps_size[0] - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1, cv2.LINE_AA)

        # Divider line
        cv2.line(image, (0, bar_h), (w, bar_h), COL_DIM, 1)

    def draw_detection_panel(self, image, prediction_text, confidence, status):
        """Draw the detection result panel near the bottom."""
        h, w, _ = image.shape
        panel_w = w - 200
        panel_h = 60
        panel_x = 15
        panel_y = h - 150

        self._overlay_rect(image, panel_x, panel_y, panel_w, panel_h, COL_BG, 0.82)

        # Pulse border when detected
        if status == "DETECTED":
            pulse = COL_PULSE_A if (self._frame_count // 8) % 2 == 0 else COL_PULSE_B
            self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, pulse, 2)
        else:
            self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, COL_DIM, 1)

        # Status dot
        if status == "DETECTED":
            dot_col = COL_GREEN
        elif status == "UNCERTAIN":
            dot_col = COL_AMBER
        else:
            dot_col = COL_GREY
        cv2.circle(image, (panel_x + 18, panel_y + 25), 6, dot_col, -1, cv2.LINE_AA)

        # Status label
        cv2.putText(image, status, (panel_x + 32, panel_y + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_col, 1, cv2.LINE_AA)

        # Prediction text
        cv2.putText(image, prediction_text, (panel_x + 18, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 1, cv2.LINE_AA)

        # Confidence bar
        bar_x = panel_x + panel_w - 260
        bar_y = panel_y + 15
        self.draw_confidence_bar(image, bar_x, bar_y, 180, 16, confidence)

    def draw_sequence_panel(self, image, sequence_text):
        """Draw the sequence text box."""
        h, w, _ = image.shape
        panel_w = w - 200
        panel_h = 40
        panel_x = 15
        panel_y = h - 85

        self._overlay_rect(image, panel_x, panel_y, panel_w, panel_h, COL_BG, 0.78)
        self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, COL_DIM, 1)

        if sequence_text:
            # Label
            cv2.putText(image, "SEQUENCE:", (panel_x + 12, panel_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_AMBER, 1, cv2.LINE_AA)
            # Text
            cv2.putText(image, sequence_text, (panel_x + 110, panel_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(image, "SEQUENCE:  (waiting for gestures...)",
                        (panel_x + 12, panel_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM, 1, cv2.LINE_AA)

    def draw_controls_bar(self, image):
        """Draw the bottom controls bar."""
        h, w, _ = image.shape
        bar_h = 30
        bar_y = h - bar_h
        self._overlay_rect(image, 0, bar_y, w, bar_h, COL_BG, 0.85)
        cv2.line(image, (0, bar_y), (w, bar_y), COL_DIM, 1)

        controls = "[Q] Quit    [C] Clear    [S] Screenshot"
        cv2.putText(image, controls, (15, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GREY, 1, cv2.LINE_AA)

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

                # FPS tracking
                now = time.time()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0:
                    self._fps_buffer.append(1.0 / dt)
                fps = np.mean(self._fps_buffer) if self._fps_buffer else 0.0

                # Flip
                if flip_image:
                    image = cv2.flip(image, 1)

                # Process with MediaPipe
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                image.flags.writeable = True

                # ── Prediction logic ────────────────────────────────
                prediction_text = "No hand detected"
                confidence = 0.0
                status = "SCANNING"

                if results.multi_hand_landmarks:
                    left_hand_landmarks = None
                    right_hand_landmarks = None

                    # Draw custom skeleton + classify left/right
                    if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            self.draw_hand_skeleton(image, hand_landmarks)

                            hand_label = handedness.classification[0].label
                            if hand_label == "Left":
                                left_hand_landmarks = hand_landmarks
                            else:
                                right_hand_landmarks = hand_landmarks

                            # Draw hand label
                            h_img, w_img, _ = image.shape
                            hand_x = int(min([lm.x for lm in hand_landmarks.landmark]) * w_img)
                            hand_y = int(min([lm.y for lm in hand_landmarks.landmark]) * h_img)
                            cv2.putText(image, hand_label,
                                        (hand_x, max(hand_y - 15, 55)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1, cv2.LINE_AA)
                    else:
                        # No handedness info — just draw
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.draw_hand_skeleton(image, hand_landmarks)

                    # Build feature vector and predict
                    features = None

                    if self.is_two_handed_model:
                        # Two-handed model
                        if left_hand_landmarks and right_hand_landmarks:
                            # Both hands clearly identified
                            features = self.preprocess_two_hands(left_hand_landmarks, right_hand_landmarks)
                        elif len(results.multi_hand_landmarks) == 2:
                            # Two hands but handedness not clearly split — use first as left, second as right
                            features = self.preprocess_two_hands(
                                results.multi_hand_landmarks[0],
                                results.multi_hand_landmarks[1]
                            )
                        elif len(results.multi_hand_landmarks) == 1:
                            # Single hand — pad with zeros (matches training format)
                            features = self.preprocess_single_hand_for_two_handed_model(
                                results.multi_hand_landmarks[0]
                            )
                    else:
                        # Single-hand model — just use first detected hand
                        features = self.preprocess_landmarks(results.multi_hand_landmarks[0])

                    if features is not None:
                        gesture, conf = self.trainer.predict(features)
                        self.history_buffer.append((gesture, conf))

                        smooth_gesture, smooth_confidence = self.get_smoothed_prediction()

                        if smooth_gesture and smooth_confidence >= self.recognition_threshold:
                            prediction_text = smooth_gesture.upper()
                            confidence = smooth_confidence
                            status = "DETECTED"
                            self.update_sequence(smooth_gesture, smooth_confidence)
                        else:
                            prediction_text = "Analysing..."
                            confidence = smooth_confidence if smooth_confidence > 0 else conf
                            status = "UNCERTAIN"

                # ── Draw HUD ────────────────────────────────────────
                sequence_text = self.get_sequence_text()

                self.draw_top_bar(image, fps)
                self.draw_detection_panel(image, prediction_text, confidence, status)
                self.draw_sequence_panel(image, sequence_text)
                self.draw_controls_bar(image)

                # Gesture legend (top-right area, below the top bar)
                h_img, w_img, _ = image.shape
                self.draw_gesture_legend(image, w_img - 175, 55)

                # Display
                cv2.imshow('Sign Language AI', image)

                # Key handling
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed.")
                    break
                elif key == ord('c'):
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