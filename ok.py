import cv2
import numpy as np

def main():
    # Open a connection to the webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initial state mode
    # Modes: 'o' = Original, 'g' = Grayscale, 'c' = Canny, 's' = Sobel
    mode = 'o'
    
    print("--- Controls ---")
    print("Press 'o' for Original Feed")
    print("Press 'g' for Grayscale Filter")
    print("Press 'c' for Canny Edge Detection")
    print("Press 's' for Sobel Edge Detection")
    print("Press 'q' to Quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Process the frame based on the active mode
        if mode == 'o':
            processed_frame = frame.copy()
            
        elif mode == 'g':
            # Grayscale Filter
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        elif mode == 'c':
            # Canny Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 100 and 200 are the lower and upper thresholds for edge linking
            processed_frame = cv2.Canny(gray, 100, 200)
            
        elif mode == 's':
            # Sobel Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Blur to reduce high-frequency background noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # Calculate gradients along X and Y axes
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            # Combine the two gradients mathematically
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Convert back to an 8-bit image
            processed_frame = np.uint8(np.clip(sobel_magnitude, 0, 255))

        # Display the resulting frame
        cv2.imshow('Real-Time Image Processing', processed_frame)

        # Wait for 1 millisecond and check if a key is pressed
        key = cv2.waitKey(1) & 0xFF
        
        # Check keys to dynamically switch modes or quit
        if key == ord('q'):
            break
        elif key in [ord('o'), ord('g'), ord('c'), ord('s')]:
            mode = chr(key)

    # Clean up and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
