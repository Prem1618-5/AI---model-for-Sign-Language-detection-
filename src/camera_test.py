"""
Camera Test Module for Sign Language Detection ML Project

Tests camera initialization, frame capture, and display.
Handles headless and non-interactive environments gracefully.
"""

import sys
import time
import argparse
import cv2


def test_camera(camera_id: int = 0, duration: float = 10.0, headless: bool = False, max_frames: int = None) -> bool:
    """
    Test camera initialization, frame acquisition, and display.
    
    Args:
        camera_id (int): Device index of the camera.
        duration (float): Maximum duration to display feed in seconds.
        headless (bool): If True, skip GUI window display.
        max_frames (int, optional): Maximum number of frames to test.
        
    Returns:
        bool: True if camera was opened and captured at least one valid frame, False otherwise.
    """
    print(f"Testing camera access (device index: {camera_id})...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera (device index: {camera_id}). "
              "Ensure camera is connected and not in use by another application.")
        return False
    
    gui_available = not headless
    frames_captured = 0
    start_time = time.time()
    
    try:
        # Test initial frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Camera opened, but failed to capture a test frame from device {camera_id}.")
            return False
        
        h, w = frame.shape[:2]
        c = frame.shape[2] if len(frame.shape) > 2 else 1
        print(f"Camera opened successfully: {w}x{h} ({c} channels) at device index {camera_id}.")
        
        if headless:
            print("Headless mode active: verifying frame capture without GUI window.")
        else:
            print(f"Showing video feed for up to {duration:.1f} seconds. Press 'q' to exit early.")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture subsequent frame during test loop.")
                break
            
            frames_captured += 1
            
            if gui_available:
                try:
                    cv2.imshow('Camera Test', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested early exit via 'q'.")
                        break
                except cv2.error:
                    print("Display server/GUI unavailable; switching to headless mode.")
                    gui_available = False
            
            # Check elapsed time
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break
            
            # Check max frame limit if set
            if max_frames is not None and frames_captured >= max_frames:
                break
        
        elapsed = time.time() - start_time
        fps = frames_captured / elapsed if elapsed > 0 else 0.0
        print(f"Camera test completed successfully: {frames_captured} frames captured in {elapsed:.2f}s (~{fps:.1f} FPS).")
        return True

    except Exception as e:
        print(f"Unexpected error during camera test: {e}")
        return False
        
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Camera resources released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Test webcam access and frame acquisition.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--frames", type=int, default=None, help="Maximum number of frames to capture")
    
    args = parser.parse_args()
    success = test_camera(
        camera_id=args.camera,
        duration=args.duration,
        headless=args.headless,
        max_frames=args.frames
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()