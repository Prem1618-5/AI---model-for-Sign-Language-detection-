"""
Adversarial Stress Test Script for M1 Preprocessing & Data Flow
Tests edge cases, degenerate data inputs, and dataset persistence integrity.
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"d:\Development Project\Sign Language"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import parse_raw_sample, GestureDataProcessor

print("=== ADVERSARIAL STRESS TESTS ===")

# Test 1: parse_raw_sample with None, strings, numbers, empty nested lists
assert parse_raw_sample(None) == [], "Failed None test"
assert parse_raw_sample([]) == [], "Failed empty list test"
assert parse_raw_sample([[]]) == [], "Failed empty inner list test"
assert parse_raw_sample([[{"x": 0.1, "y": 0.2, "z": 0.3}], []]) == [[{"x": 0.1, "y": 0.2, "z": 0.3}]], "Failed ragged inner list test"

# Test 2: Preprocessing pipeline on existing dataset in data/raw
processor = GestureDataProcessor(
    data_dir=os.path.join(PROJECT_ROOT, "data", "raw"),
    processed_dir=os.path.join(PROJECT_ROOT, "data", "processed")
)

gesture_data, is_two_handed = processor.load_gesture_data()
print(f"Loaded raw gestures: {list(gesture_data.keys())}, is_two_handed: {is_two_handed}")

# Total raw samples across all gestures
total_raw_samples = sum(len(samples) for samples in gesture_data.values())
print(f"Total raw samples across all files: {total_raw_samples}")

# Check prepare_dataset output
X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(augment=True)
total_processed = len(X_train) + len(X_val) + len(X_test)
print(f"Total processed samples (with 5x augmentation): {total_processed}")

# Raw sample count: 350. Each sample produces 1 original + 5 augmentations = 6 samples. 350 * 6 = 2100 samples.
assert total_processed == total_raw_samples * 6, f"Expected {total_raw_samples * 6} samples, got {total_processed}"
assert X_train.shape[1] == 126, f"Expected 126 features, got {X_train.shape[1]}"

# Verify non-trivial data distribution (no all-zero vectors or NaNs)
assert not np.isnan(X_train).any(), "NaNs found in X_train"
assert not np.isinf(X_train).any(), "Infs found in X_train"
# Ensure features are not static / hardcoded constants
variances = np.var(X_train, axis=0)
assert (variances[:63] > 0).all(), "Zero variance found in primary hand features"

# Verify NPZ file saved correctly
npz_file = os.path.join(PROJECT_ROOT, "data", "processed", "processed_gesture_data.npz")
assert os.path.exists(npz_file), "NPZ file does not exist"
loaded_npz = np.load(npz_file)
assert "X_train" in loaded_npz
assert "y_train" in loaded_npz
assert "class_names" in loaded_npz

print("\nALL ADVERSARIAL STRESS INVARIANTS CONFIRMED!")
