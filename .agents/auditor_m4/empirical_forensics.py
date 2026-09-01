import os
import sys
import numpy as np
import tensorflow as tf

# Add src to sys.path
src_dir = r"d:\Development Project\Sign Language\src"
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_preprocessing import GestureDataProcessor, parse_raw_sample
from model_training import GestureModelTrainer
from temporal_filter import TemporalSmoother
from ui_overlay import SignLanguageHUD, HUDState

print("=== 1. Forensic Math Normalization Invariant Check ===")
processor = GestureDataProcessor()

# Test synthetic hand
raw_hand = [{'x': float(i)*0.01 + 0.5, 'y': float(i)*0.02 + 0.3, 'z': float(i)*0.005, 'visibility': 1.0} for i in range(21)]

# 1. Normalization
norm_hand = processor.normalize_landmarks(raw_hand)
pts = np.array([[p['x'], p['y'], p['z']] for p in norm_hand])
wrist = pts[0]
middle_mcp = pts[9]
palm_center = (wrist + middle_mcp) / 2
print(f"Palm center after normalization (should be ~0,0,0): {palm_center}")
assert np.allclose(palm_center, [0, 0, 0], atol=1e-6), "Palm center not origin!"

dist = np.linalg.norm(middle_mcp - wrist)
print(f"Wrist-to-middle MCP distance after normalization (should be 1.0): {dist:.6f}")
assert np.isclose(dist, 1.0, atol=1e-5), "Unit distance scale violated!"

# Translation invariance
translated_hand = [{'x': p['x'] + 50.0, 'y': p['y'] - 20.0, 'z': p['z'] + 10.0, 'visibility': 1.0} for p in raw_hand]
norm_trans = processor.normalize_landmarks(translated_hand)
pts_trans = np.array([[p['x'], p['y'], p['z']] for p in norm_trans])
diff_trans = np.max(np.abs(pts - pts_trans))
print(f"Max diff under translation: {diff_trans:.8f}")
assert diff_trans < 1e-6, "Translation invariance violated!"

# Scale invariance
scaled_hand = [{'x': (p['x']-0.5)*3.5 + 0.5, 'y': (p['y']-0.3)*3.5 + 0.3, 'z': p['z']*3.5, 'visibility': 1.0} for p in raw_hand]
norm_scaled = processor.normalize_landmarks(scaled_hand)
pts_scaled = np.array([[p['x'], p['y'], p['z']] for p in norm_scaled])
diff_scaled = np.max(np.abs(pts - pts_scaled))
print(f"Max diff under scaling: {diff_scaled:.8f}")
assert diff_scaled < 1e-5, "Scale invariance violated!"

# Zero-division guard on degenerate hand
degen_hand = [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0} for _ in range(21)]
norm_degen = processor.normalize_landmarks(degen_hand)
assert len(norm_degen) == 21, "Degenerate hand handling failed!"
print("Math Normalization: 100% PASS")

print("\n=== 2. Forensic Neural Network Direct Tensor Inference Check ===")
trainer = GestureModelTrainer()
model = trainer.build_model(input_shape=(63,), num_classes=5)
dummy_input = np.random.randn(1, 63).astype(np.float32)

# Verify direct tensor execution
tensor_in = tf.convert_to_tensor(dummy_input, dtype=tf.float32)
out = model(tensor_in, training=False).numpy()
print(f"Model output shape: {out.shape}, sum of probs: {np.sum(out):.6f}")
assert out.shape == (1, 5)
assert np.isclose(np.sum(out), 1.0, atol=1e-5), "Softmax sum != 1.0"

# Verify trainer.predict_proba
trainer.class_names = ["hello", "thank_you", "yes", "no", "good"]
probs = trainer.predict_proba(dummy_input[0])
print(f"predict_proba output shape: {probs.shape}, sum: {np.sum(probs):.6f}")
assert probs.shape == (5,)
assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

pred_class, conf = trainer.predict(dummy_input[0])
print(f"predict result: class={pred_class}, conf={conf:.4f}")
assert pred_class in trainer.class_names
assert 0.0 <= conf <= 1.0
print("Tensor Inference: 100% PASS")

print("\n=== 3. Forensic Temporal Filter Check (EMA, Hysteresis, Velocity Gating) ===")
smoother = TemporalSmoother(
    alpha=0.25,
    threshold_high=0.80,
    threshold_low=0.45,
    debounce_frames=4,
    velocity_threshold=0.08,
    sequence_timeout=2.0,
    class_names=["HELLO", "WORLD"]
)

# Step 1: Initial state is SCANNING
assert smoother.current_state == "SCANNING"

# Step 2: Feed high confidence on HELLO for 3 frames (debounce is 4 -> should be UNCERTAIN / Analysing)
for i in range(3):
    res_class, res_conf, res_status = smoother.update([0.95, 0.05], wrist_pos=(0.5, 0.5), timestamp=float(i)*0.033)
    assert res_status == "UNCERTAIN", f"Frame {i} should be UNCERTAIN, got {res_status}"
    assert res_class == "Analysing..."

# Step 3: Frame 4 triggers DETECTED
res_class, res_conf, res_status = smoother.update([0.95, 0.05], wrist_pos=(0.5, 0.5), timestamp=4.0*0.033)
assert res_status == "DETECTED", f"Frame 4 should be DETECTED, got {res_status}"
assert res_class == "HELLO"
print("Hysteresis Debouncing: PASS")

# Step 4: Confidence dips to 0.60 (above threshold_low=0.45) -> Should STAY in DETECTED
res_class, res_conf, res_status = smoother.update([0.55, 0.45], wrist_pos=(0.5, 0.5), timestamp=5.0*0.033)
assert res_status == "DETECTED", f"Frame 5 should remain DETECTED, got {res_status}"
assert res_class == "HELLO"
print("Hysteresis Hold above T_low: PASS")

# Step 5: Velocity spike -> Should gate to UNCERTAIN
res_class, res_conf, res_status = smoother.update([0.95, 0.05], wrist_pos=(0.8, 0.9), timestamp=6.0*0.033) # large wrist movement
assert res_status == "UNCERTAIN", f"Velocity jump should trigger UNCERTAIN, got {res_status}"
print("Kinematic Velocity Gating: PASS")

# Step 6: Sequence timeout
smoother.last_update_time = 10.0
smoother.last_gesture_time = 10.0
smoother.update([0.1, 0.9], wrist_pos=(0.5, 0.5), timestamp=15.0) # > 2.0s gap
assert len(smoother.get_sequence()) == 0, "Sequence timeout failed to clear!"
print("Inactivity Timeout: PASS")

print("\n=== 4. Forensic In-Place ROI Alpha Blending Check ===")
hud = SignLanguageHUD()
img = np.zeros((720, 1280, 3), dtype=np.uint8)
img[:, :] = 100 # base gray

# Blend ROI
hud.blend_roi(img, 100, 100, 200, 100, colour=(15, 15, 30), alpha=0.8)
# Outside ROI should remain 100
assert np.all(img[50, 50] == 100), "Outside ROI corrupted!"
# Inside ROI should be modified in-place
roi_val = img[150, 150]
expected_val = (np.array([15, 15, 30]) * 0.8 + np.array([100, 100, 100]) * 0.2).astype(np.uint8)
assert np.allclose(roi_val, expected_val, atol=2), f"ROI blend math mismatch: {roi_val} vs {expected_val}"
print("ROI Alpha Compositing: PASS")

print("\n=== 5. Dependency Audit ===")
req_path = r"d:\Development Project\Sign Language\requirements.txt"
with open(req_path, "r") as f:
    req_lines = [line.strip().lower() for line in f if line.strip() and not line.startswith('#')]

print("Requirements.txt contents:")
for r in req_lines:
    print(f"  - {r}")

assert not any("pandas" in r for r in req_lines), "Unapproved dependency 'pandas' found in requirements.txt!"
assert not any("seaborn" in r for r in req_lines), "Unapproved dependency 'seaborn' found in requirements.txt!"

print("Dependency Audit: 100% PASS (pandas and seaborn successfully pruned)")
