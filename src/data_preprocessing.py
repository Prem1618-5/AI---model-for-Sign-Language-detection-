"""
Data Preprocessing Module for Sign Language Detection ML Project

This module handles the preprocessing of collected hand landmark data,
including normalization, augmentation, and preparation for model training.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class GestureDataProcessor:
    """
    Handles preprocessing of hand gesture landmark data.
    
    Attributes:
        data_dir (str): Directory containing raw gesture data files
        processed_dir (str): Directory to save processed data
        random_seed (int): Random seed for reproducibility
        label_encoder (LabelEncoder): Encoder for gesture labels
    """
    
    def __init__(self, data_dir='../data/raw', processed_dir='../data/processed', random_seed=42):
        """
        Initialize the GestureDataProcessor with specified parameters.
        
        Args:
            data_dir (str): Directory containing raw gesture data files
            processed_dir (str): Directory to save processed data
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_gesture_data(self, file_pattern='*.json'):
        """
        Load all gesture data files from the data directory.
        
        Args:
            file_pattern (str): Pattern to match gesture data files
            
        Returns:
            tuple: (gesture_data, is_two_handed) where gesture_data is a dictionary with gesture
                  names as keys and lists of landmark samples as values, and is_two_handed is a
                  boolean indicating if the dataset contains two-handed gestures
        """
        gesture_data = {}
        is_two_handed = False
        
        # Find all gesture files
        data_files = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        if not data_files:
            raise ValueError(f"No gesture data files found in {self.data_dir}")
        
        print(f"Found {len(data_files)} gesture data files.")
        
        # Load each file
        for file_path in tqdm(data_files, desc="Loading gesture data"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            gesture_name = data['gesture_name']
            landmarks = data['landmarks']
            
            # Check if this is two-handed data format
            if 'two_hands' in data and data['two_hands']:
                is_two_handed = True
            
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = []
            
            gesture_data[gesture_name].extend(landmarks)
        
        print("Loaded gesture data:")
        for gesture, samples in gesture_data.items():
            print(f"  - {gesture}: {len(samples)} samples")
        
        if is_two_handed:
            print("Detected two-handed gesture data format")
        
        return gesture_data, is_two_handed
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize hand landmarks to make them invariant to scale and translation.
        
        Args:
            landmarks (list): List of hand landmark points
            
        Returns:
            list: Normalized landmark points
        """
        # Convert to numpy array for easier manipulation
        points = np.array([[p['x'], p['y'], p['z']] for p in landmarks])
        
        # Calculate center of palm (average of wrist and middle finger MCP)
        wrist = points[0]
        middle_mcp = points[9]  # Middle finger MCP joint
        palm_center = (wrist + middle_mcp) / 2
        
        # Translate points to make palm center the origin
        centered_points = points - palm_center
        
        # Scale to make the distance from wrist to middle finger MCP = 1
        scale_reference = np.linalg.norm(middle_mcp - wrist)
        if scale_reference > 0:
            normalized_points = centered_points / scale_reference
        else:
            normalized_points = centered_points
        
        # Convert back to list of dictionaries with the same keys
        normalized_landmarks = []
        for i, (px, py, pz) in enumerate(normalized_points):
            normalized_landmarks.append({
                'x': float(px),
                'y': float(py),
                'z': float(pz),
                'visibility': landmarks[i].get('visibility', 1.0)
            })
        
        return normalized_landmarks
    
    def flatten_landmarks(self, landmarks):
        """
        Flatten landmarks into a 1D array suitable for ML models.
        
        Args:
            landmarks (list): List of landmark dictionaries
            
        Returns:
            numpy.ndarray: Flattened array of landmark coordinates
        """
        # Extract x, y, z coordinates
        flattened = []
        for lm in landmarks:
            flattened.extend([lm['x'], lm['y'], lm['z']])
        
        return np.array(flattened)
    
    def augment_landmarks(self, landmarks, num_augmentations=5):
        """
        Generate augmented versions of hand landmarks for better training.
        
        Args:
            landmarks (list): List of landmark dictionaries
            num_augmentations (int): Number of augmented versions to generate
            
        Returns:
            list: List of augmented landmark sets
        """
        augmented_sets = []
        
        for _ in range(num_augmentations):
            # Convert to numpy array
            points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
            
            # Apply random rotation around z-axis (2D rotation in x-y plane)
            theta = np.random.uniform(-0.2, 0.2)  # Small rotation
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            rotated_points = np.dot(points, rotation_matrix)
            
            # Apply small random translations
            translation = np.random.uniform(-0.1, 0.1, size=3)
            translated_points = rotated_points + translation
            
            # Apply small random scaling
            scale = np.random.uniform(0.9, 1.1)
            scaled_points = translated_points * scale
            
            # Convert back to list of dictionaries
            augmented_landmarks = []
            for i, (px, py, pz) in enumerate(scaled_points):
                augmented_landmarks.append({
                    'x': float(px),
                    'y': float(py),
                    'z': float(pz),
                    'visibility': landmarks[i].get('visibility', 1.0)
                })
            
            augmented_sets.append(augmented_landmarks)
        
        return augmented_sets
    
    def prepare_dataset(self, augment=True, test_size=0.2, val_size=0.1):
        """
        Prepare the full dataset for training, including normalization, augmentation, and train/val/test split.
        
        Args:
            augment (bool): Whether to perform data augmentation
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
        """
        # Load raw gesture data
        gesture_data, is_two_handed = self.load_gesture_data()
        
        # Prepare lists for processed data
        X = []
        y = []
        class_names = list(gesture_data.keys())
        
        # Encode class labels
        encoded_labels = self.label_encoder.fit_transform(class_names)
        label_dict = dict(zip(class_names, encoded_labels))
        
        print("Processing and normalizing landmarks...")
        
        # Process each gesture
        for gesture_name, samples in tqdm(gesture_data.items(), desc="Processing gestures"):
            label = label_dict[gesture_name]
            
            # Process each sample
            for sample in samples:
                if is_two_handed:
                    # Two-handed format: sample is a list of hands
                    if len(sample) == 1:
                        # Only one hand was detected - normalize and flatten it
                        normalized = self.normalize_landmarks(sample[0])
                        flattened = self.flatten_landmarks(normalized)
                        X.append(flattened)
                        y.append(label)
                    elif len(sample) == 2:
                        # Both hands were detected - normalize and combine them
                        normalized_hand1 = self.normalize_landmarks(sample[0])
                        normalized_hand2 = self.normalize_landmarks(sample[1])
                        
                        # Flatten each hand separately then concatenate
                        flattened_hand1 = self.flatten_landmarks(normalized_hand1)
                        flattened_hand2 = self.flatten_landmarks(normalized_hand2)
                        
                        # Combine features from both hands
                        combined_features = np.concatenate([flattened_hand1, flattened_hand2])
                        X.append(combined_features)
                        y.append(label)
                        
                        # If augmenting, create augmented versions
                        if augment:
                            # Augment each hand separately
                            augmented_hand1_sets = self.augment_landmarks(normalized_hand1)
                            augmented_hand2_sets = self.augment_landmarks(normalized_hand2)
                            
                            # Combine augmented hands (match them randomly)
                            for i in range(min(len(augmented_hand1_sets), len(augmented_hand2_sets))):
                                aug_hand1 = augmented_hand1_sets[i]
                                aug_hand2 = augmented_hand2_sets[i]
                                
                                flat_aug_hand1 = self.flatten_landmarks(aug_hand1)
                                flat_aug_hand2 = self.flatten_landmarks(aug_hand2)
                                
                                combined_aug = np.concatenate([flat_aug_hand1, flat_aug_hand2])
                                X.append(combined_aug)
                                y.append(label)
                else:
                    # Single-handed format (original)
                    # Normalize landmarks
                    normalized = self.normalize_landmarks(sample)
                    
                    # Flatten to feature vector
                    flattened = self.flatten_landmarks(normalized)
                    X.append(flattened)
                    y.append(label)
                    
                    # If augmenting, create augmented versions
                    if augment:
                        augmented_sets = self.augment_landmarks(normalized)
                        for aug_sample in augmented_sets:
                            X.append(self.flatten_landmarks(aug_sample))
                            y.append(label)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Processed dataset: {X.shape[0]} samples, {X.shape[1]} features")
        if is_two_handed:
            print(f"Using two-handed features with {X.shape[1]} dimensions")
        
        # Split into train+val and test sets
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        # Split train+val into train and val sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, 
            test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
            random_state=self.random_seed,
            stratify=y_trainval
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Save processed data
        metadata = {
            'is_two_handed': is_two_handed,
            'feature_dim': X.shape[1]
        }
        self.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, class_names, metadata)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names
    
    def save_processed_data(self, X_train, y_train, X_val, y_val, X_test, y_test, class_names, metadata):
        """
        Save processed dataset to disk.
        
        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Split datasets
            class_names (list): List of class names
            metadata (dict): Metadata about the dataset
        """
        # Create a dictionary with all data
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': class_names,
            'feature_dim': metadata['feature_dim'],
            'num_classes': len(class_names),
            'is_two_handed': metadata['is_two_handed']
        }
        
        # Save to numpy compressed format
        np.savez_compressed(
            os.path.join(self.processed_dir, 'processed_gesture_data.npz'),
            **data_dict
        )
        
        # Also save class names separately for easy access
        with open(os.path.join(self.processed_dir, 'class_names.json'), 'w') as f:
            json.dump(class_names, f)
        
        print(f"Saved processed data to {self.processed_dir}")
    
    def load_processed_data(self):
        """
        Load processed dataset from disk.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names, is_two_handed)
        """
        processed_file = os.path.join(self.processed_dir, 'processed_data.npz')
        
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found: {processed_file}")
        
        # Load data from npz file
        data = np.load(processed_file, allow_pickle=True)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        class_names = data['class_names']
        
        # Check if this is a two-handed dataset
        is_two_handed = bool(data.get('is_two_handed', False))
        
        print(f"Loaded processed data:")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Validation samples: {X_val.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Feature dimension: {X_train.shape[1]}")
        print(f"  - Number of classes: {len(class_names)}")
        if is_two_handed:
            print(f"  - Two-handed dataset: Yes")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names, is_two_handed

if __name__ == "__main__":
    # Example usage
    processor = GestureDataProcessor()
    
    # Check if processed data already exists
    processed_data_path = os.path.join(processor.processed_dir, 'processed_gesture_data.npz')
    
    if os.path.exists(processed_data_path):
        print("Loading previously processed data...")
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, is_two_handed = processor.load_processed_data()
        print("Data loaded successfully.")
    else:
        print("Processing raw gesture data...")
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=True)
    
    # Display dataset stats
    print("\nDataset Statistics:")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Validation set: {X_val.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Feature dimension: {X_train.shape[1]}")
    print(f"  - Classes: {class_names}") 