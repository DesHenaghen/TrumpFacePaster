# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2


def open_camera():
    cap = cv2.VideoCapture(0)

    if (not cap.isOpened()):
        print "ERROR: Error opening camera"
        exit(-1)

    read_input(cap)


def read_input(cap):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        print("Found {0} faces!".format(len(faces)))

        # Draw a Donald Trump face over the faces
        for (ox, oy, w, h) in faces:
            # Load a random Donald Trump face
            face_image = cv2.imread("donald_face-1.png", -1)
            # Resize the face to fit the face detected
            face = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_CUBIC)

            # Draw the Donald face over the detected face
            for c in range(0, 3):
                height = face.shape[0]
                width = face.shape[1]
                alpha = face[:, :, 3] / 255.0
                color = face[:, :, c] * (alpha)
                beta = frame[oy:oy + height, ox:ox + width, c] * (1.0 - alpha)

                frame[oy:oy + height, ox:ox + width, c] = color + beta

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Main Method
if __name__ == "__main__":
    open_camera()
