import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread('C:\\Users\\aadit\\OneDrive\\Documents\\Python Projects\\pythonw\\ai.py\\Image_manipulation\\AiPicnic.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, f'People Count: {len(faces)}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Face Tracking and Counting', image)


    
        
cv2.destroyAllWindows()