import os
import cv2
import dlib
import numpy as np

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the face recognition model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(
    'dlib_face_recognition_resnet_model_v1.dat')

# Load the images of the known people
known_embeddings = []
for img_path in os.listdir('raw_faces'):
    if img_path.endswith('.jpg'):
        img = cv2.imread(os.path.join('raw_faces', img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img, 1)
        if len(faces) > 0:
            shape = predictor(img, faces[0])
            embedding = np.array(facerec.compute_face_descriptor(img, shape))
            known_embeddings.append(embedding)
# Open a video capture device
cap = cv2.VideoCapture(0)

# Process each frame of the video capture device
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    # Recognize each face and label it
    for face in faces:
        shape = predictor(frame, face)
        embedding = np.array(facerec.compute_face_descriptor(frame, shape))

        # Compute the distances between the face embedding and the known embeddings
        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # Label the face as the known person with the smallest distance
        if min_distance < 0.6:
            name = 'Rain'
        else:
            name = 'Unknown'

        # Draw a rectangle around the face and label it
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the labeled frame
    cv2.imshow('dupa', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy the window
cap.release()
cv2.destroyAllWindows()
