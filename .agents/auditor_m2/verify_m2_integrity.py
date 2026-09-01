"""
Milestone M2 Forensic Integrity Verification Script
Independently authored by Forensic Auditor M2.
Tests and verifies:
1. TemporalSmoother genuine EMA math, dual hysteresis, velocity gating, no mock sequences.
2. GestureModelTrainer genuine TensorFlow inference (sensitivity to weights, direct tensor evaluation, probability conservation).
3. Data preprocessing zero data leakage (split before augmentation, disjoint train/val/test sets).
4. Complete dependency audit (zero seaborn, zero pandas, pure matplotlib confusion matrix).
"""

import os
import sys
import ast
import json
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from temporal_filter import TemporalSmoother
from model_training import GestureModelTrainer
from data_preprocessing import GestureDataProcessor, parse_raw_sample

def run_audits():
    results = {}

    print("=" * 70)
    print("M2 FORENSIC INTEGRITY AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. TEMPORAL FILTER INTEGRITY & MATHEMATICAL VERIFICATION
    # ---------------------------------------------------------
    print("\n[CHECK 1] TemporalSmoother Mathematical & Algorithmic Verification...")
    
    # 1.1 AST Check for Hardcoded/Facade Patterns
    tf_path = os.path.join(SRC_DIR, 'temporal_filter.py')
    with open(tf_path, 'r', encoding='utf-8') as f:
        tf_source = f.read()
    
    tree = ast.parse(tf_source)
    # Check for hardcoded mock returns or stubs
    hardcoded_strings = ["mock_gesture", "fixed_output", "return 'hello'", "return \"hello\""]
    found_mock = any(mock in tf_source for mock in hardcoded_strings)
    results['temporal_filter_ast_clean'] = not found_mock
    print(f"  - AST clean / no mock tokens: {not found_mock}")

    # 1.2 EMA Step Response & Mathematical Invariant
    alpha = 0.25
    smoother = TemporalSmoother(alpha=alpha, threshold_high=0.80, threshold_low=0.45, class_names=['A', 'B', 'C'])
    
    # Step input: initial [1, 0, 0], then continuous [0, 1, 0]
    smoother.update(np.array([1.0, 0.0, 0.0]))
    expected_s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    step_input = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    ema_math_exact = True
    for step in range(1, 10):
        expected_s = alpha * step_input + (1.0 - alpha) * expected_s
        expected_s = expected_s / np.sum(expected_s)
        smoother.update(step_input)
        if not np.allclose(smoother.smoothed_probs, expected_s, atol=1e-5):
            ema_math_exact = False
            print(f"  - EMA mismatch at step {step}: got {smoother.smoothed_probs}, expected {expected_s}")
            break
    
    results['ema_exact_math'] = ema_math_exact
    print(f"  - EMA exact analytical step response match: {ema_math_exact}")

    # 1.3 Dual-Threshold Hysteresis State Machine Verification
    smoother.reset()
    # Step A: 3 frames of high confidence (0.85) -> debounce_frames is 4 -> should be UNCERTAIN
    for i in range(3):
        pred, conf, status = smoother.update(np.array([0.85, 0.10, 0.05]))
        assert status == "UNCERTAIN", f"Frame {i+1} should be UNCERTAIN"
    
    # Frame 4 -> transitions to DETECTED
    pred, conf, status = smoother.update(np.array([0.85, 0.10, 0.05]))
    assert status == "DETECTED" and pred == "A", f"Frame 4 should transition to DETECTED, got {status}, {pred}"
    
    # Frame 5 -> drop to 0.50 (between T_low=0.45 and T_high=0.80) -> MUST STAY DETECTED
    pred, conf, status = smoother.update(np.array([0.50, 0.30, 0.20]))
    assert status == "DETECTED" and pred == "A", f"Hysteresis failed to hold DETECTED above T_low, got {status}"
    
    # Drop below T_low=0.45 -> should transition to SCANNING
    for _ in range(5):
        pred, conf, status = smoother.update(np.array([0.20, 0.40, 0.40]))
    assert status == "SCANNING" and pred == "None", f"Expected drop to SCANNING, got {status}, {pred}"
    results['hysteresis_state_machine'] = True
    print("  - Dual-threshold hysteresis state machine verified: PASS")

    # 1.4 Kinematic Velocity Gating Verification
    smoother.reset()
    # Steady hand
    t = 100.0
    for _ in range(5):
        t += 0.033
        pred, conf, status = smoother.update(np.array([0.90, 0.05, 0.05]), wrist_pos=(0.5, 0.5), timestamp=t)
    assert status == "DETECTED"
    
    # Fast displacement (wrist jumps from 0.5, 0.5 to 0.9, 0.9 in 33ms) -> displacement 0.565 >> 0.08
    t += 0.033
    pred, conf, status = smoother.update(np.array([0.90, 0.05, 0.05]), wrist_pos=(0.9, 0.9), timestamp=t)
    assert status == "UNCERTAIN", f"Kinematic velocity gating failed, got {status}"
    results['velocity_gating'] = True
    print("  - Kinematic wrist velocity gating verified: PASS")

    # ---------------------------------------------------------
    # 2. GESTURE MODEL TRAINER GENUINE TENSORFLOW INFERENCE
    # ---------------------------------------------------------
    print("\n[CHECK 2] GestureModelTrainer Genuine Neural Inference Verification...")
    
    trainer = GestureModelTrainer(model_dir='models')
    model = trainer.build_model(input_shape=(63,), num_classes=3)
    trainer.class_names = ['alpha', 'beta', 'gamma']

    # 2.1 Verify direct tensor inference vs weights
    test_sample_1d = np.ones((63,), dtype=np.float32)
    test_batch_2d = np.ones((4, 63), dtype=np.float32)

    probs_1d = trainer.predict_proba(test_sample_1d)
    probs_2d = trainer.predict_proba(test_batch_2d)

    assert probs_1d.shape == (3,), f"Expected shape (3,), got {probs_1d.shape}"
    assert probs_2d.shape == (4, 3), f"Expected shape (4, 3), got {probs_2d.shape}"
    assert np.isclose(np.sum(probs_1d), 1.0, atol=1e-5), f"Probabilities do not sum to 1: {np.sum(probs_1d)}"
    assert np.allclose(np.sum(probs_2d, axis=1), np.ones(4), atol=1e-5)

    # Alter weights explicitly to force class 2 (gamma) activation
    # Final dense layer is layer index -1
    final_dense = model.layers[-1]
    weights, biases = final_dense.get_weights()
    new_weights = np.zeros_like(weights)
    new_weights[:, 2] = 10.0  # Force huge logits for class 2
    new_biases = np.zeros_like(biases)
    new_biases[2] = 5.0
    final_dense.set_weights([new_weights, new_biases])

    altered_probs = trainer.predict_proba(test_sample_1d)
    pred_class, conf = trainer.predict(test_sample_1d)
    
    # Verify inference output responds dynamically to neural weights
    weights_evaluated_dynamically = (pred_class == 'gamma') and (altered_probs[2] > 0.99)
    results['neural_weights_evaluated_genuinely'] = weights_evaluated_dynamically
    print(f"  - Model dynamically evaluates neural network tensor weights: {weights_evaluated_dynamically} (class={pred_class}, p_gamma={altered_probs[2]:.6f})")

    # ---------------------------------------------------------
    # 3. DATA LEAKAGE VERIFICATION
    # ---------------------------------------------------------
    print("\n[CHECK 3] Data Preprocessing & Training Leakage Verification...")

    # Inspect prepare_dataset code structure
    dp_path = os.path.join(SRC_DIR, 'data_preprocessing.py')
    with open(dp_path, 'r', encoding='utf-8') as f:
        dp_source = f.read()

    # Verify split order: train_test_split must precede augmentation loop
    split_pos = dp_source.find("train_test_split(indices")
    if split_pos == -1:
        split_pos = dp_source.find("train_test_split")
    aug_pos = dp_source.find("augment_landmarks")
    
    # Check that in prepare_dataset, splitting happens before data augmentation
    split_before_aug = (split_pos != -1 and aug_pos != -1 and split_pos < aug_pos)
    results['split_before_augmentation_in_ast'] = split_before_aug
    print(f"  - train_test_split executed before augmentation loop: {split_before_aug}")

    # Run dataset loading to verify disjoint sets
    processor = GestureDataProcessor(data_dir=os.path.join(PROJECT_ROOT, 'data', 'raw'))
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, is_two_handed = processor.load_processed_data(
        os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_gesture_data.npz')
    )
    
    # Check shape consistency
    print(f"  - Dataset shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    # Check for exact feature duplicate overlap between unaugmented val/test and train
    test_in_train = 0
    val_in_train = 0
    for test_sample in X_test:
        matches = np.all(np.isclose(X_train, test_sample, atol=1e-6), axis=1)
        if np.any(matches):
            test_in_train += 1

    for val_sample in X_val:
        matches = np.all(np.isclose(X_train, val_sample, atol=1e-6), axis=1)
        if np.any(matches):
            val_in_train += 1

    zero_leakage = (test_in_train == 0 and val_in_train == 0)
    results['zero_data_leakage'] = zero_leakage
    print(f"  - Zero data leakage into validation or test sets: {zero_leakage} (val_overlap={val_in_train}, test_overlap={test_in_train})")

    # ---------------------------------------------------------
    # 4. DEPENDENCY AUDIT & PONYTAIL COMPLIANCE
    # ---------------------------------------------------------
    print("\n[CHECK 4] Dependency & Ponytail Compliance Verification...")
    
    # Check requirements.txt
    req_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        req_text = f.read().lower()
    
    has_seaborn = 'seaborn' in req_text
    has_pandas = 'pandas' in req_text
    results['seaborn_removed_from_requirements'] = not has_seaborn
    results['pandas_removed_from_requirements'] = not has_pandas
    print(f"  - seaborn absent from requirements.txt: {not has_seaborn}")
    print(f"  - pandas absent from requirements.txt: {not has_pandas}")

    # Check all python files in src/ for seaborn/pandas imports
    prohibited_imports_found = []
    for root, dirs, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                f_tree = ast.parse(content)
                for node in ast.walk(f_tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            if n.name in ['seaborn', 'pandas']:
                                prohibited_imports_found.append((file, n.name))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in ['seaborn', 'pandas']:
                            prohibited_imports_found.append((file, node.module))

    results['prohibited_imports_in_src'] = prohibited_imports_found
    print(f"  - Prohibited imports in src/: {prohibited_imports_found} (Count: {len(prohibited_imports_found)})")

    # Save evidence json
    evidence_path = os.path.join(os.path.dirname(__file__), 'audit_evidence.json')
    with open(evidence_path, 'w', encoding='utf-8') as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 70)
    all_passed = (
        results['temporal_filter_ast_clean'] and
        results['ema_exact_math'] and
        results['hysteresis_state_machine'] and
        results['velocity_gating'] and
        results['neural_weights_evaluated_genuinely'] and
        results['split_before_augmentation_in_ast'] and
        results['zero_data_leakage'] and
        results['seaborn_removed_from_requirements'] and
        results['pandas_removed_from_requirements'] and
        len(prohibited_imports_found) == 0
    )
    verdict = "CLEAN" if all_passed else "INTEGRITY VIOLATION"
    print(f"OVERALL VERDICT: {verdict}")
    print("=" * 70)
    return verdict

if __name__ == '__main__':
    verdict = run_audits()
    sys.exit(0 if verdict == "CLEAN" else 1)
