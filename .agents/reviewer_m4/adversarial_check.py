"""
Adversarial Stress Test & Integrity Audit Script for Milestone M4 Review
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from data_preprocessing import GestureDataProcessor, parse_raw_sample
from temporal_filter import TemporalSmoother
from ui_overlay import SignLanguageHUD, HUDState

def test_adversarial_preprocessing():
    print("Testing Preprocessing Edge Cases...")
    proc = GestureDataProcessor()
    
    # 1. Empty sample
    assert parse_raw_sample([]) == [], "Failed on empty list"
    assert parse_raw_sample([[]]) == [], "Failed on nested empty list"
    
    # 2. Degenerate hand (all zeros)
    zero_hand = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
    norm_zero = proc.normalize_landmarks(zero_hand)
    assert len(norm_zero) == 21, "Failed on zero hand normalization length"
    assert all(p['x'] == 0.0 and p['y'] == 0.0 and p['z'] == 0.0 for p in norm_zero), "Failed zero hand centering"
    
    # 3. Degenerate identical wrist and MCP (zero distance)
    ident_hand = [{'x': 0.5, 'y': 0.5, 'z': 0.5} for _ in range(21)]
    norm_ident = proc.normalize_landmarks(ident_hand)
    assert len(norm_ident) == 21, "Failed on identical wrist/MCP"
    assert all(p['x'] == 0.0 and p['y'] == 0.0 and p['z'] == 0.0 for p in norm_ident), "Failed identical wrist/MCP centering"
    
    # 4. Flattening dimensions
    flat63 = proc.flatten_landmarks(norm_zero)
    assert flat63.shape == (63,), f"Expected (63,), got {flat63.shape}"
    
    print("  -> Preprocessing edge cases PASSED.")

def test_adversarial_temporal_filter():
    print("Testing Temporal Filter Adversarial Scenarios...")
    classes = ["hello", "no", "thanks", "yes"]
    smoother = TemporalSmoother(
        alpha=0.25,
        threshold_high=0.80,
        threshold_low=0.45,
        debounce_frames=4,
        velocity_threshold=0.08,
        sequence_timeout=2.0,
        class_names=classes
    )
    
    # Scenario A: Rapidly oscillating noisy predictions below threshold
    # Should stay in SCANNING
    for i in range(20):
        noisy_probs = np.random.uniform(0.1, 0.4, size=4)
        noisy_probs /= noisy_probs.sum()
        pred, conf, status = smoother.update(noisy_probs, timestamp=float(i)*0.033)
        assert status == "SCANNING", f"Expected SCANNING on noise, got {status}"
        assert pred == "None", f"Expected None on noise, got {pred}"
    
    # Scenario B: High confidence burst but fewer than debounce_frames (e.g. 2 frames)
    # Should enter UNCERTAIN (Analysing...), not DETECTED
    smoother.reset()
    high_probs = np.array([0.95, 0.02, 0.02, 0.01])
    pred1, conf1, status1 = smoother.update(high_probs, timestamp=1.0)
    pred2, conf2, status2 = smoother.update(high_probs, timestamp=1.033)
    assert status1 == "UNCERTAIN", f"Expected UNCERTAIN on frame 1, got {status1}"
    assert status2 == "UNCERTAIN", f"Expected UNCERTAIN on frame 2, got {status2}"
    assert pred1 == "Analysing..."
    
    # Scenario C: 4 consecutive high frames -> enters DETECTED
    pred3, conf3, status3 = smoother.update(high_probs, timestamp=1.066)
    pred4, conf4, status4 = smoother.update(high_probs, timestamp=1.100)
    assert status4 == "DETECTED", f"Expected DETECTED on frame 4, got {status4}"
    assert pred4 == "hello", f"Expected hello, got {pred4}"
    
    # Scenario D: Hysteresis hold: drop confidence to 0.55 (> T_low 0.45, < T_high 0.80)
    # Should stay DETECTED
    mid_probs = np.array([0.55, 0.15, 0.15, 0.15])
    pred5, conf5, status5 = smoother.update(mid_probs, timestamp=1.133)
    assert status5 == "DETECTED", f"Expected DETECTED to hold in hysteresis, got {status5}"
    assert pred5 == "hello"
    
    # Scenario E: Kinematic Velocity Gate
    # Sudden large wrist jump (> velocity threshold) -> drops to UNCERTAIN
    pred6, conf6, status6 = smoother.update(high_probs, wrist_pos=(0.1, 0.1), timestamp=1.166)
    pred7, conf7, status7 = smoother.update(high_probs, wrist_pos=(0.9, 0.9), timestamp=1.200)
    assert status7 == "UNCERTAIN", f"Expected UNCERTAIN on rapid movement, got {status7}"
    assert pred7 == "Analysing..."
    
    # Scenario F: Inactivity Timeout
    # Advance time by 3.0 seconds (> sequence_timeout 2.0s) -> clears sequence and state
    pred8, conf8, status8 = smoother.update(noisy_probs, timestamp=5.0)
    assert len(smoother.sequence_buffer) == 0, "Sequence buffer should have cleared after timeout"
    assert status8 == "SCANNING"
    
    print("  -> Temporal filter adversarial scenarios PASSED.")

def test_adversarial_hud():
    print("Testing HUD Rendering Adversarial Inputs...")
    hud = SignLanguageHUD(class_names=["hello", "no", "thanks", "yes"])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 1. Out of bounds ROI coordinates (negative, exceeding frame, flipped)
    hud.blend_roi(frame, -100, -100, 50, 50)
    hud.blend_roi(frame, 1500, 1000, 200, 200)
    hud.blend_roi(frame, 500, 500, -50, -50)
    hud.blend_roi(frame, 100, 100, 0, 0)
    
    # 2. Extreme HUDState inputs
    state = HUDState(
        fps=-999.0,
        detected=True,
        status_text="CORRUPT_STATE_TEST",
        gesture="SUPER_LONG_GESTURE_NAME_THAT_EXCEEDS_NORMAL_BOUNDS_XYZ_1234567890",
        confidence=1.99,
        sequence=["HELLO", "WORLD", "GESTURE", "A", "B", "C", "D", "E", "F", "G", "H"],
        hands_info=[
            {'bbox': (-50, -50, 1500, 1000), 'landmarks': None, 'handedness': 'Alien', 'score': float('nan')},
            {'bbox': (0, 0, 10, 10), 'landmarks': None, 'handedness': None, 'score': None}
        ],
        classes=["A", "B", "C", "D"],
        sequence_progress=-0.5
    )
    
    rendered = hud.render(frame, state)
    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)
    
    # 3. Empty HUDState
    empty_state = HUDState()
    rendered_empty = hud.render(frame, empty_state)
    assert rendered_empty is not None
    assert rendered_empty.shape == (720, 1280, 3)
    
    print("  -> HUD rendering adversarial inputs PASSED.")

if __name__ == "__main__":
    test_adversarial_preprocessing()
    test_adversarial_temporal_filter()
    test_adversarial_hud()
    print("\nALL ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY!")
