import cv2
import dlib
import imutils
from imutils import face_utils

# Load the pre-trained model
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Load an image
image_path = "path_to_your_image.jpg"
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
rects = detector(gray, 1)

# Loop over the face detections
for (i, rect) in enumerate(rects):
    # Determine the facial landmarks for the face region
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Convert dlib's rectangle to a bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Loop over the (x, y)-coordinates for the facial landmarks
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# Display the output image with the face detections and landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
