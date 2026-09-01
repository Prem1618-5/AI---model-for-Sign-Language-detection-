"""
Data Preprocessing Module for Sign Language Detection ML Project

This module handles the preprocessing of collected hand landmark data,
including normalization, augmentation, and preparation for model training.
"""

import os
import json
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_raw_sample(sample: list) -> list:
    """
    Normalizes any raw sample representation into a list of hands:
    - Single hand flat: [ {x, y, z}, ... 21 dicts ] -> [ [ {x, y, z}, ... 21 dicts ] ]
    - Multi-hand list: [ [ {x, y, z}, ... 21 dicts ], ... ] -> preserved as-is
    
    Args:
        sample (list): Raw landmark data representing either a flat hand or list of hands.
        
    Returns:
        list: Normalized list of hands, where each hand is a list of landmark dicts.
    """
    if not sample:
        return []
    if isinstance(sample[0], dict):
        # Single hand flat list of 21 landmark dictionaries
        return [sample]
    elif isinstance(sample[0], (list, tuple)):
        # Multi-hand list where each item is a list of landmark dicts
        return [list(h) for h in sample if isinstance(h, (list, tuple)) and len(h) > 0]
    return [sample]


class GestureDataProcessor:
    """
    Handles preprocessing of hand gesture landmark data.
    
    Attributes:
        data_dir (str): Directory containing raw gesture data files
        processed_dir (str): Directory to save processed data
        random_seed (int): Random seed for reproducibility
        label_encoder (LabelEncoder): Encoder for gesture labels
    """
    
    parse_raw_sample = staticmethod(parse_raw_sample)
    
    def __init__(self, data_dir='data/raw', processed_dir='data/processed', random_seed=42):
        """
        Initialize the GestureDataProcessor with specified parameters.
        
        Args:
            data_dir (str): Directory containing raw gesture data files
            processed_dir (str): Directory to save processed data or specific .npz file path
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        
        # Support passing either a directory or a specific .npz file path
        if processed_dir.endswith('.npz'):
            self.processed_file = processed_dir
            self.processed_dir = os.path.dirname(processed_dir) or '.'
        else:
            self.processed_file = None
            self.processed_dir = processed_dir
            if os.path.exists(processed_dir) and not os.path.isdir(processed_dir):
                raise FileExistsError(f"Target path '{processed_dir}' exists and is not a directory.")
            os.makedirs(self.processed_dir, exist_ok=True)
    
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
            
            # Check if this is two-handed data format from metadata or sample content
            if data.get('two_hands', False):
                is_two_handed = True
            
            for s in landmarks:
                parsed = parse_raw_sample(s)
                if len(parsed) >= 2:
                    is_two_handed = True
                    break
            
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
        Prepare the full dataset for training, including normalization, clean training-only
        augmentation (zero test/val data leakage), and stratified train/val/test split.
        
        Args:
            augment (bool): Whether to perform data augmentation on training set
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
        """
        # Load raw gesture data
        gesture_data, is_two_handed = self.load_gesture_data()
        
        class_names = list(gesture_data.keys())
        
        # Encode class labels
        encoded_labels = self.label_encoder.fit_transform(class_names)
        label_dict = dict(zip(class_names, encoded_labels))
        
        print("Processing and normalizing landmarks...")
        
        # Collect raw samples and unaugmented features
        raw_samples = []  # list of dicts with hands, label, features
        
        for gesture_name, samples in tqdm(gesture_data.items(), desc="Processing gestures"):
            label = label_dict[gesture_name]
            
            for sample in samples:
                hands = parse_raw_sample(sample)
                if not hands:
                    continue
                
                if is_two_handed:
                    if len(hands) == 1:
                        normalized = self.normalize_landmarks(hands[0])
                        flattened = self.flatten_landmarks(normalized)
                        features = np.concatenate([flattened, np.zeros(63)])
                    else:
                        normalized_hand1 = self.normalize_landmarks(hands[0])
                        normalized_hand2 = self.normalize_landmarks(hands[1])
                        flattened_hand1 = self.flatten_landmarks(normalized_hand1)
                        flattened_hand2 = self.flatten_landmarks(normalized_hand2)
                        features = np.concatenate([flattened_hand1, flattened_hand2])
                else:
                    normalized = self.normalize_landmarks(hands[0])
                    features = self.flatten_landmarks(normalized)
                
                raw_samples.append({
                    'hands': hands,
                    'label': label,
                    'features': features
                })
        
        if not raw_samples:
            raise ValueError("No valid landmark samples found in dataset.")
        
        X_all = np.array([s['features'] for s in raw_samples])
        y_all = np.array([s['label'] for s in raw_samples])
        indices = np.arange(len(raw_samples))
        
        print(f"Base dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
        if is_two_handed:
            print(f"Using two-handed features with {X_all.shape[1]} dimensions")
        
        # Split unaugmented indices to guarantee zero data leakage into val and test
        idx_trainval, idx_test, y_trainval, y_test = train_test_split(
            indices, y_all, test_size=test_size, random_state=self.random_seed, stratify=y_all
        )
        
        idx_train, idx_val, y_train_base, y_val = train_test_split(
            idx_trainval, y_trainval,
            test_size=val_size / (1.0 - test_size),
            random_state=self.random_seed,
            stratify=y_trainval
        )
        
        X_test = X_all[idx_test]
        X_val = X_all[idx_val]
        
        # Build training set with augmentation applied ONLY to training indices
        X_train_list = []
        y_train_list = []
        
        for idx in idx_train:
            item = raw_samples[idx]
            hands = item['hands']
            label = item['label']
            base_features = item['features']
            
            X_train_list.append(base_features)
            y_train_list.append(label)
            
            if augment:
                if is_two_handed:
                    if len(hands) == 1:
                        normalized = self.normalize_landmarks(hands[0])
                        augmented_hand_sets = self.augment_landmarks(normalized)
                        for aug_hand in augmented_hand_sets:
                            flat_aug = self.flatten_landmarks(aug_hand)
                            padded_aug = np.concatenate([flat_aug, np.zeros(63)])
                            X_train_list.append(padded_aug)
                            y_train_list.append(label)
                    else:
                        normalized_hand1 = self.normalize_landmarks(hands[0])
                        normalized_hand2 = self.normalize_landmarks(hands[1])
                        aug_sets1 = self.augment_landmarks(normalized_hand1)
                        aug_sets2 = self.augment_landmarks(normalized_hand2)
                        for i in range(min(len(aug_sets1), len(aug_sets2))):
                            flat1 = self.flatten_landmarks(aug_sets1[i])
                            flat2 = self.flatten_landmarks(aug_sets2[i])
                            combined_aug = np.concatenate([flat1, flat2])
                            X_train_list.append(combined_aug)
                            y_train_list.append(label)
                else:
                    normalized = self.normalize_landmarks(hands[0])
                    augmented_sets = self.augment_landmarks(normalized)
                    for aug_sample in augmented_sets:
                        X_train_list.append(self.flatten_landmarks(aug_sample))
                        y_train_list.append(label)
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        print(f"Train set: {X_train.shape[0]} samples (augmented={augment})")
        print(f"Validation set: {X_val.shape[0]} samples (clean, unaugmented)")
        print(f"Test set: {X_test.shape[0]} samples (clean, unaugmented)")
        
        # Save processed data
        metadata = {
            'is_two_handed': is_two_handed,
            'feature_dim': X_train.shape[1]
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
    
    def load_processed_data(self, processed_file=None):
        """
        Load processed dataset from disk.
        
        Args:
            processed_file (str, optional): Explicit path to .npz file
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names, is_two_handed)
        """
        target_file = processed_file or self.processed_file or os.path.join(self.processed_dir, 'processed_gesture_data.npz')
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Processed data file not found: {target_file}")
        
        # Load data from npz file
        data = np.load(target_file, allow_pickle=True)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        class_names = list(data['class_names'])
        
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