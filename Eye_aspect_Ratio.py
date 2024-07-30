import face_recognition
import cv2

# Load an image
image_path = "images/neutral.jpg"
image = face_recognition.load_image_file(image_path)

# Find all facial landmarks in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Convert to OpenCV format
image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Loop over each face's landmarks
for face_landmarks in face_landmarks_list:
    for feature, points in face_landmarks.items():
        for (x, y) in points:
            cv2.circle(image_cv2, (x, y), 2, (0, 255, 0), -1)

# Display the output image with the face landmarks
cv2.imshow("Output", image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
