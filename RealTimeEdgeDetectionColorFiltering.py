import cv2
import numpy as np

def apply_filter(image, ftype):
    img = image.copy()
    if ftype == "red tint":
        img[:, :, 0] = 0  # Set blue channel to 0
        img[:, :, 1] = 0  # Set green channel to 0
    elif ftype == "green tint":
        img[:, :, 0] = 0  # Set blue channel to 0
        img[:, :, 2] = 0  # Set red channel to 0
    elif ftype == "blue tint":
        img[:, :, 1] = 0  # Set green channel to 0
        img[:, :, 2] = 0  # Set red channel to 0
    elif ftype == "sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sob = cv2.bitwise_or(sobelx.astype('uint8'), sobely.astype('uint8'))
        img = cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)
    elif ftype == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        can = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR) 
    elif ftype == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2. THRESH_BINARY, 9, 9)
        color = cv2.bilaterialfilter(image, 9, 300, 300)
        img = cv2.bitwise_and(color, color, mask=edges)
    return img
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    ftype = "original"
    print("Keys: r-Red. g-Green. b-Blue. s-Sobel. c-Canny. t-Cartoon. q-Quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        out = apply_filter(frame, ftype)
        cv2.imshow("Filtered", out)
        key = v2.waitKey(1) & 0xFF
        if key == ord('r'):
            ftype = "red_tint"
        elif key == ord('g'):
            ftype = "green_tint"
        elif key == ord('b'):
            ftype = "blue_tint"
        elif key == ord('s'):
            ftype = "sobel"
        elif key == ord('c'):
            ftype = "canny"
        elif key == ord('t'):
            ftype = "cartoon"
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()      
