import cv2
import mediapipe as mp
import os
import numpy as np


# Pngs fold 
base_path = r"C:\\"




# Person name
person = input("Nombre Completo_Rut (Ej. JuanRodriguez): ")
save_path = os.path.join(base_path, person)
os.makedirs(save_path, exist_ok=True) # crate fold if not exists

count = 0 # imagen count

# mediapipe variables process
face_mesh = mp.solutions.face_mesh 
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# video capture with face points
cap = cv2.VideoCapture(0)   # 0 = principal camera, 1 = second camera.. etc
if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

with face_mesh.FaceMesh(    # face_mesh: defining variables
    max_num_faces = 10,
    refine_landmarks = True,
    min_detection_confidence = 0.8,
    min_tracking_confidence = 0.8
) as face_mesh:

    while True: # if its true...
        ret, frame = cap.read() # ret can be false or true, about signal; frame capture camera in array
        if not ret:
            print("No ha llegado la señal")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #face_mesh work with rgb and cv2 with bgr
        results = face_mesh.process(rgb_frame) # results with rgb

        if results.multi_face_landmarks: # drawing points on face
            for face_landmarks in results.multi_face_landmarks:
                # landmarks to pixels
                h, w, c = frame.shape
                coords = np.array([(int(p.x*w), int(p.y*h)) for p in face_landmarks.landmark])

                # Bounding box minima
                x_min, y_min = coords.min(axis = 0)
                x_max, y_max = coords.max(axis = 0)

                # margen
                pad_top = 45
                pad_bot = 20
                x_min = max(x_min - pad_bot, 0)
                y_min = max(y_min - pad_top, 0)
                x_max = min(x_max + pad_bot, w)
                y_max = min(y_max + pad_bot, h)

                face = frame[y_min:y_max, x_min:x_max]

                
                # saving face with 'shift + S'

                key = cv2.waitKey(1) & 0xFF
                if key == ord('S'):
                    file_name = os.path.join(save_path, f"{person}_{count}.png")
                    cv2.imwrite(file_name, face)
                    print(f"Se ha guardado: {file_name}")
                    count += 1

                # Drawing boxes and points togheter
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                drawing.draw_landmarks(frame, face_landmarks)
        
        cv2.putText(frame, f"Presionar 'Shift' + 'S' para guardar rostro",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 100, 200), 1)


        cv2.imshow("Camera_face_mesh", frame)

        if cv2.waitKey(1) % 0xFF == ord("q") or cv2.getWindowProperty("Camera_face_mesh", cv2.WND_PROP_VISIBLE) < 1:
            print("Se ha cerrado")
            break

cap.release()

cv2.destroyAllWindows()
