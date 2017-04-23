import cv2
import sys
from random import randint

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a Donald Trump face over the faces
for (ox, oy, w, h) in faces:
    # Load a random Donald Trump face
    face_image = cv2.imread("donald_face-" + str(randint(1, 6)) + ".png", -1)
    # Resize the face to fit the face detected
    face = cv2.resize(face_image,(w, h), interpolation = cv2.INTER_CUBIC)

    # Draw the Donald face over the detected face
    for c in range(0,3):
        height = face.shape[0]
        width = face.shape[1]
        alpha = face[:, :, 3] / 255.0
        color = face[:, :, c] * (alpha)
        beta  = image[oy:oy + height, ox:ox + width, c] * (1.0-alpha)

        image[oy:oy + height, ox:ox + width, c] = color + beta

cv2.imshow("Faces found", image)
cv2.waitKey(0)