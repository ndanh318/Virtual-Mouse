import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = face_mesh.process(gray)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                print(f"Landmark {idx + 1}: x={x}, y={y}, z={z}")

                # Vẽ tọa độ (x, y) lên ảnh (có thể sử dụng cv2.circle để vẽ đồng thời)
                cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (cx, cy), radius=1, color=(0, 255, 0), thickness=-1)

    cv2.imshow('Face Mesh Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
