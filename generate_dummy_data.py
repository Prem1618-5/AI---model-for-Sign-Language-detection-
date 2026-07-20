import os
import json
import numpy as np
from datetime import datetime

def generate_dummy_dataset():
    # Set output directory
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)

    gestures = ["hello", "yes", "no"]
    num_samples = 50

    for gesture in gestures:
        landmarks_data = []
        for _ in range(num_samples):
            # Generate 21 landmarks for a single hand
            hand_landmarks = []
            for i in range(21):
                if i == 0:
                    # Wrist
                    hand_landmarks.append({'x': 0.5, 'y': 0.8, 'z': 0.0, 'visibility': 1.0})
                elif i == 9:
                    # Middle finger MCP
                    hand_landmarks.append({'x': 0.5, 'y': 0.4, 'z': 0.0, 'visibility': 1.0})
                else:
                    # Random offsets
                    rx = 0.5 + np.random.uniform(-0.2, 0.2)
                    ry = 0.6 + np.random.uniform(-0.2, 0.2)
                    rz = np.random.uniform(-0.05, 0.05)
                    hand_landmarks.append({'x': rx, 'y': ry, 'z': rz, 'visibility': 1.0})
            
            # For single hand format, the sample is the hand_landmarks list itself
            landmarks_data.append(hand_landmarks)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        gesture_data = {
            'gesture_name': gesture,
            'timestamp': timestamp,
            'num_samples': len(landmarks_data),
            'landmarks': landmarks_data,
            'two_hands': False
        }
        
        with open(filepath, 'w') as f:
            json.dump(gesture_data, f, indent=2)
            
        print(f"Generated dummy data for {gesture} -> {filepath}")

if __name__ == "__main__":
    generate_dummy_dataset()
