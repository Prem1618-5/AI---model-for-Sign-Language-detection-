"""
Real-time Recognition Module for Sign Language Detection ML Project

This module handles real-time gesture recognition using a webcam,
allowing for interactive sign language detection.
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

from data_preprocessing import GestureDataProcessor
from model_training import GestureModelTrainer

class RealtimeGestureRecognizer:
    """
    Handles real-time gesture recognition using webcam input.
    
    Attributes:
        model_path (str): Path to the trained model
        min_detection_confidence (float): Minimum confidence for hand detection
        min_tracking_confidence (float): Minimum confidence for hand tracking
        recognition_threshold (float): Minimum confidence threshold for gesture recognition
        smoothing_window (int): Window size for temporal smoothing
        history_buffer (deque): Buffer for storing recent predictions
        sequence_buffer (list): Buffer for storing gesture sequences
        sequence_timeout (float): Timeout for sequence detection in seconds
        last_gesture_time (float): Timestamp of last recognized gesture
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
            max_num_hands=2,  # Track up to 2 hands
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
        self.sequence_timeout = 2.0  # seconds
        self.last_gesture_time = 0
        
        # Processor for landmark normalization
        self.processor = GestureDataProcessor()
        
        # Flag to track two-handed gestures
        self.is_two_handed_model = False  # Will be set based on model metadata
    
    def preprocess_landmarks(self, landmarks, is_left_hand=None):
        """
        Preprocess hand landmarks for model input.
        
        Args:
            landmarks (list): MediaPipe hand landmarks
            is_left_hand (bool): Whether the hand is the left hand
            
        Returns:
            numpy.ndarray: Processed features for model input
        """
        # Convert MediaPipe landmarks to our format
        landmark_list = []
        for landmark in landmarks.landmark:
            landmark_list.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': 1.0
            })
        
        # Normalize landmarks
        normalized = self.processor.normalize_landmarks(landmark_list)
        
        # Convert to feature vector
        features = self.processor.flatten_landmarks(normalized)
        
        return features
    
    def preprocess_two_hands(self, left_hand_landmarks, right_hand_landmarks):
        """
        Preprocess landmarks from both hands together.
        
        Args:
            left_hand_landmarks (list): MediaPipe landmarks for left hand
            right_hand_landmarks (list): MediaPipe landmarks for right hand
            
        Returns:
            numpy.ndarray: Combined processed features for model input
        """
        # Process each hand
        left_features = self.preprocess_landmarks(left_hand_landmarks, True) if left_hand_landmarks else np.zeros(63)
        right_features = self.preprocess_landmarks(right_hand_landmarks, False) if right_hand_landmarks else np.zeros(63)
        
        # Combine features
        combined_features = np.concatenate([left_features, right_features])
        
        return combined_features
    
    def get_smoothed_prediction(self):
        """
        Get smoothed prediction based on recent history.
        
        Returns:
            tuple: (gesture_name, confidence) or (None, 0) if no clear prediction
        """
        if not self.history_buffer:
            return None, 0.0
        
        # Count occurrences of each gesture
        gesture_counts = {}
        gesture_confidences = {}
        
        for gesture, confidence in self.history_buffer:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                gesture_confidences[gesture] = 0.0
            
            gesture_counts[gesture] += 1
            gesture_confidences[gesture] += confidence
        
        # Find most frequent gesture
        max_count = 0
        max_gesture = None
        
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                max_gesture = gesture
        
        # Check if dominant gesture appears in more than 60% of the window
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
        
        # Check if we should start a new sequence (timeout exceeded)
        if current_time - self.last_gesture_time > self.sequence_timeout:
            self.sequence_buffer = []
        
        # Update last gesture time
        self.last_gesture_time = current_time
        
        # Only add to sequence if it's different from the last one
        if not self.sequence_buffer or self.sequence_buffer[-1][0] != gesture:
            self.sequence_buffer.append((gesture, confidence))
            
            # Limit sequence length
            if len(self.sequence_buffer) > 10:
                self.sequence_buffer.pop(0)
    
    def get_sequence_text(self):
        """
        Get current gesture sequence as text.
        
        Returns:
            str: Space-separated gesture sequence
        """
        return ' '.join([item[0] for item in self.sequence_buffer])
    
    def run(self, camera_id=0, flip_image=True):
        """
        Run real-time gesture recognition with webcam feed.
        
        Args:
            camera_id (int): Camera device ID
            flip_image (bool): Whether to flip the camera image horizontally
        """
        # Check if model and class names are loaded
        if self.model is None:
            raise ValueError("Model not loaded. Provide a valid model path.")
        
        if not self.class_names:
            raise ValueError("Class names not available. Check model metadata.")
        
        print("Starting real-time gesture recognition...")
        print(f"Loaded {len(self.class_names)} gestures: {', '.join(self.class_names)}")
        print(f"Camera ID: {camera_id}, Flip image: {flip_image}")
        print("Press 'q' to quit, 'c' to clear sequence, 'r' to resize window")
        
        # Initialize webcam
        try:
            print(f"Attempting to open camera {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera with ID {camera_id}")
            
            # Set camera properties for better resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("Camera opened successfully.")
            print(f"Camera properties: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
        except Exception as e:
            print(f"Error opening camera: {e}")
            return
        
        try:
            # Set up display window
            print("Creating display window...")
            cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Sign Language Recognition', 1280, 720)
            
            print("Starting main loop...")
            # Main loop
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error reading from camera")
                    # Wait a moment and try again
                    time.sleep(0.1)
                    continue
                
                # Flip the image horizontally for a more intuitive mirror view
                if flip_image:
                    image = cv2.flip(image, 1)
                
                # To improve performance, optionally mark the image as not writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image)
                
                # Draw the hand annotations
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                prediction_text = "No hand detected"
                confidence_text = ""
                
                left_hand_landmarks = None
                right_hand_landmarks = None
                
                if results.multi_hand_landmarks:
                    # Get hand landmarks and classify as left or right when available
                    if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
                        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Identify left/right hand
                            hand_label = handedness.classification[0].label
                            if hand_label == "Left":
                                left_hand_landmarks = hand_landmarks
                            elif hand_label == "Right":
                                right_hand_landmarks = hand_landmarks
                            
                            # Add label to image
                            hand_x = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
                            hand_y = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
                            cv2.putText(image, hand_label, (hand_x, hand_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Process hand landmarks for prediction
                    if len(results.multi_hand_landmarks) == 2 and left_hand_landmarks and right_hand_landmarks:
                        # Two hands detected - use combined features
                        features = self.preprocess_two_hands(left_hand_landmarks, right_hand_landmarks)
                        gesture, confidence = self.trainer.predict(features)
                        
                        # Add to history buffer
                        self.history_buffer.append((gesture, confidence))
                        
                        # Get smoothed prediction
                        smooth_gesture, smooth_confidence = self.get_smoothed_prediction()
                        
                        if smooth_gesture and smooth_confidence >= self.recognition_threshold:
                            prediction_text = f"Detected: {smooth_gesture}"
                            confidence_text = f"Confidence: {smooth_confidence:.2f}"
                            
                            # Update sequence
                            self.update_sequence(smooth_gesture, smooth_confidence)
                        else:
                            prediction_text = "Uncertain gesture"
                            confidence_text = f"Confidence: {smooth_confidence:.2f}"
                    
                    elif len(results.multi_hand_landmarks) == 1:
                        # Single hand detected
                        features = self.preprocess_landmarks(results.multi_hand_landmarks[0])
                        gesture, confidence = self.trainer.predict(features)
                        
                        # Add to history buffer
                        self.history_buffer.append((gesture, confidence))
                        
                        # Get smoothed prediction
                        smooth_gesture, smooth_confidence = self.get_smoothed_prediction()
                        
                        if smooth_gesture and smooth_confidence >= self.recognition_threshold:
                            prediction_text = f"Detected: {smooth_gesture}"
                            confidence_text = f"Confidence: {smooth_confidence:.2f}"
                            
                            # Update sequence
                            self.update_sequence(smooth_gesture, smooth_confidence)
                        else:
                            prediction_text = "Uncertain gesture"
                            confidence_text = f"Confidence: {smooth_confidence:.2f}"
                
                # Get sequence text
                sequence_text = self.get_sequence_text()
                
                # Add text to the image
                cv2.putText(image, prediction_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, confidence_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw a background rectangle for the sequence text
                if sequence_text:
                    text_size = cv2.getTextSize(sequence_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(image, (10, image.shape[0] - 50), 
                                 (10 + text_size[0], image.shape[0] - 10), 
                                 (0, 0, 0), -1)
                    cv2.putText(image, sequence_text, (10, image.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display the image
                cv2.imshow('Sign Language Recognition', image)
                
                # Handle key presses
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed.")
                    break
                elif key == ord('c'):
                    # Clear the sequence buffer
                    self.sequence_buffer = []
                    print("Sequence cleared")
                elif key == ord('r'):
                    # Let the user resize the window
                    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Press 'r' again when done resizing. Current size: {current_width}x{current_height}")
        
        except Exception as e:
            print(f"Error in recognition loop: {e}")
        finally:
            # Clean up
            print("Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            print("Recognition stopped.")

if __name__ == "__main__":
    # Example usage
    # Check if a model exists in the models directory
    print("Starting standalone real-time recognition...")
    models_dir = '../models'
    model_metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    if os.path.exists(model_metadata_path):
        # Initialize the recognizer with default model path
        print("Found model metadata, initializing recognizer...")
        recognizer = RealtimeGestureRecognizer()
        
        # Run real-time recognition
        try:
            print("Starting recognition...")
            recognizer.run()
        except Exception as e:
            print(f"Error during recognition: {e}")
    else:
        print("No trained model found. Please train a model first using model_training.py")
        print("You can use data_collection.py to collect gesture data, then process and train a model.") 