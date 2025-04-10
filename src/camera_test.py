import cv2
import time

def test_camera():
    print("Testing camera access...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully. Showing video feed for 10 seconds.")
    print("Press 'q' to exit early.")
    
    # Set start time
    start_time = time.time()
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display frame
        cv2.imshow('Camera Test', frame)
        
        # Check for key press or timeout
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or (time.time() - start_time) > 10:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test complete.")

if __name__ == "__main__":
    test_camera() 