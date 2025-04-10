"""
Data Collection Module for Sign Language Detection ML Project

This module handles the collection of hand gesture data using camera input
and MediaPipe for hand landmark detection.
"""

import os
import cv2
import time
import json
import numpy as np
import mediapipe as mp
from datetime import datetime
from tqdm import tqdm

class DataCollector:
    """
    Handles the collection of hand gesture data using computer vision.
    
    Attributes:
        output_dir (str): Directory to save collected data
        mp_hands (mediapipe.solutions.hands): MediaPipe Hands solution
        hands (mediapipe.solutions.hands.Hands): Hand detector instance
        mp_drawing (mediapipe.solutions.drawing_utils): MediaPipe drawing utilities
        capture_delay (float): Delay between frame captures in seconds
        min_detection_confidence (float): Minimum confidence for hand detection
        min_tracking_confidence (float): Minimum confidence for hand tracking
    """
    
    def __init__(self, 
                 output_dir=r'D:\Programs\Cursor.ai\Sign Language detection system using AI\sign_language_ml\data\raw',
                 capture_delay=0.2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize the DataCollector with specified parameters.
        
        Args:
            output_dir (str): Directory to save collected data
            capture_delay (float): Delay between frame captures in seconds
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
        """
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Capture parameters
        self.capture_delay = capture_delay
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
    def collect_gesture_data(self, gesture_name, num_samples=100):
        """
        Collect hand gesture data from webcam.
        
        Args:
            gesture_name (str): Name of the gesture to collect
            num_samples (int): Number of samples to collect
            
        Returns:
            str: Path to the saved gesture data file
        """
        print(f"Preparing to collect {num_samples} samples for gesture: {gesture_name}")
        print("Press 'SPACE' to start collection, 'Q' to quit")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        
        # Variables for collection process
        samples_collected = 0
        landmarks_data = []
        start_collection = False
        last_capture_time = 0
        
        # Configure camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create resizable window
        cv2.namedWindow('Hand Gesture Data Collection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Gesture Data Collection', 1280, 720)
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    # Wait a moment and try again instead of breaking
                    time.sleep(0.1)
                    continue
                
                # Flip the frame horizontally for a more intuitive display
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and get hand landmarks
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Display status information on the frame
                status_text = f"Gesture: {gesture_name} | "
                if start_collection:
                    status_text += f"Collecting: {samples_collected}/{num_samples}"
                else:
                    status_text += "Press SPACE to start"
                
                cv2.putText(
                    frame, 
                    status_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Display the frame
                cv2.imshow('Hand Gesture Data Collection', frame)
                
                # Capture landmarks if collection has started
                if start_collection and (time.time() - last_capture_time) > self.capture_delay:
                    if results.multi_hand_landmarks:
                        # Store all detected hands (up to 2)
                        frame_landmarks = []
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract landmark coordinates for each hand
                            hand_data = []
                            for lm in hand_landmarks.landmark:
                                hand_data.append({
                                    'x': lm.x,
                                    'y': lm.y,
                                    'z': lm.z,
                                    'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                                })
                            
                            frame_landmarks.append(hand_data)
                        
                        # Add to collection - now storing data for both hands when available
                        landmarks_data.append(frame_landmarks)
                        samples_collected += 1
                        last_capture_time = time.time()
                        
                        # Check if we've collected enough samples
                        if samples_collected >= num_samples:
                            break
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space key
                    start_collection = True
                    last_capture_time = time.time()
        
        except Exception as e:
            print(f"Error during data collection: {e}")
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            
            # Always try to save any data that was collected
            if landmarks_data:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{gesture_name}_{timestamp}.json"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    gesture_data = {
                        'gesture_name': gesture_name,
                        'timestamp': timestamp,
                        'num_samples': len(landmarks_data),
                        'landmarks': landmarks_data,
                        'two_hands': True  # Indicate that this data contains possibly multiple hands per frame
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(gesture_data, f, indent=2)
                    
                    print(f"Saved {len(landmarks_data)} samples to {filepath}")
                    return filepath
                except Exception as e:
                    print(f"Error saving data: {e}")
                    return None
            else:
                print("No data collected.")
                return None
    
    def collect_multiple_gestures(self, gestures_config):
        """
        Collect data for multiple gestures.
        
        Args:
            gestures_config (list): List of dictionaries with 'name' and 'samples' keys
            
        Returns:
            list: Paths to the saved gesture data files
        """
        saved_files = []
        
        for gesture in gestures_config:
            gesture_name = gesture['name']
            num_samples = gesture.get('samples', 100)
            
            print(f"\n{'='*40}")
            print(f"Collecting data for gesture: {gesture_name}")
            print(f"{'='*40}\n")
            
            filepath = self.collect_gesture_data(gesture_name, num_samples)
            if filepath:
                saved_files.append(filepath)
            
            # Small delay between gestures
            time.sleep(1)
        
        return saved_files

def extract_landmarks_from_file(filepath):
    """
    Extract landmark data from a saved JSON file.
    
    Args:
        filepath (str): Path to the gesture data JSON file
        
    Returns:
        tuple: (gesture_name, landmarks_array)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    gesture_name = data['gesture_name']
    landmarks = data['landmarks']
    
    # Convert to numpy array for efficient processing
    landmarks_array = np.array([
        [
            [point['x'], point['y'], point['z']] 
            for point in sample
        ] 
        for sample in landmarks
    ])
    
    return gesture_name, landmarks_array

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Define gestures to collect
    gestures = [
        {'name': 'hello', 'samples': 50},
        {'name': 'thank_you', 'samples': 50},
        {'name': 'yes', 'samples': 50},
        {'name': 'no', 'samples': 50},
        {'name': 'What', 'samples': 50},
        {'name': 'is', 'samples': 50},
        {'name': 'name', 'samples': 50},
        {'name': 'your', 'samples': 50},
        {'name': 'My', 'samples': 50},
        {'name': 'Good', 'samples': 50},
        {'name': 'I love you', 'samples': 50},
        {'name': 'Sorry', 'samples': 50}
    ]
    
    # Collect data for all gestures
    saved_files = collector.collect_multiple_gestures(gestures)
    
    print("\nData collection completed!")
    print(f"Collected data for {len(saved_files)} gestures.") 