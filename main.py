import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import cv2
import mediapipe as mp
import numpy as np

# ---------- Helper functions ----------

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points, landmarks, w, h):
    coords = []
    for i in eye_points:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        coords.append((x, y))

    v1 = euclidean_distance(coords[1], coords[5])
    v2 = euclidean_distance(coords[2], coords[4])
    h_dist = euclidean_distance(coords[0], coords[3])

    return (v1 + v2) / (2.0 * h_dist)

# ---------- Setup ----------

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20

# ---------- Main loop ----------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "AWAKE 👀"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESHOLD:
            status = "SLEEP 😴"

    cv2.putText(
        frame,
        status,
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4
    )

    cv2.imshow("Sleep Detection - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
