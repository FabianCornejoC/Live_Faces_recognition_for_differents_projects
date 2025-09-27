import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import mediapipe as mp
import os
import tempfile
import uuid


# -- Configuracion -- 
embeddings_file = input("Ruta del archivo embeddings (.npy): ").strip()
embeddings_file = embeddings_file.replace("\\", "/")

# Validation of existence of file
if not os.path.isfile(embeddings_file):
    raise FileNotFoundError(f"No se encontró el archivo: {embeddings_file}")

# cargando embedding
embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()


# Parameters
THRESHOLD = 0.4       # angle of arrive of cosine for considering match
PROCESS_EVERY = 8     # Frame rate
FRAME_SCALE = 0.5     # Frame size %, reduce for aceleration cpu

# Initiation mediapipe and facedetection

mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection= mp_face.FaceDetection(model_selection = 1, min_detection_confidence = 0.7)

# open camera
cap = cv2.VideoCapture(0)

frame_counter = 0
face_cache = {}


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Reducing resolution for aceleratig
    small_frame = cv2.resize(frame, (0,0), fx = FRAME_SCALE, fy=FRAME_SCALE)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Process at N frames
    if frame_counter % PROCESS_EVERY == 0:
        results = face_detection.process(rgb_frame)
        face_cache.clear() # se limpian los cache para recalcular cada embeddings, mantener si quiero identidad fija

        if results.detections:
            for idx, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = small_frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # adjust limits

                x, y = max(0, x), max(0, y)
                w, h = min(w, iw - x), min(h, ih - y)

                # rescale bbox to original size
                x_orig = int(x / FRAME_SCALE)
                y_orig = int(y / FRAME_SCALE)
                w_orig = int(w / FRAME_SCALE)
                h_orig = int(h / FRAME_SCALE)

                face_img = frame[y_orig:y_orig + h_orig, x_orig: x_orig + w_orig]

                try:
                    # generate embedding
                    tmp_name = f"C:/Users/Kim/AppData/Local/Temp/{uuid.uuid4()}.jpg" # si hay error de acceso a la carpeta, así se obliga a entrar
                    cv2.imwrite(tmp_name, face_img)

                    embeddings_list = DeepFace.represent(
                        img_path = tmp_name,
                        model_name = "Facenet",
                        enforce_detection = False
                    )
                    os.remove(tmp_name) # eliminar dps de usar
                    
                    
                    if embeddings_list and len(embeddings_list) > 0:
                        face_embedding = np.array(embeddings_list[0]["embedding"])

                        # compare with save embeddings

                        best_match_name = "Desconocido"
                        best_match_dist = 1.0

                        for person_name, emb_list in embeddings_dict.items():
                            for emb in emb_list:
                                dist = cosine(face_embedding, emb)
                                if dist < best_match_dist:
                                    best_match_dist = dist
                                    best_match_name = person_name

                        if best_match_dist > THRESHOLD:
                            best_match_name = "Desconocido"

                        # guardando cache

                        face_cache[idx] = (best_match_name, face_embedding)

                except Exception as e:
                    print(f"No se pudo procesar el rostro: {e}")

    if 'results' in locals() and results.detections:
        for idx, (name, _) in face_cache.items():
            detection = results.detections[idx]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            cv2.rectangle(frame, (x,y), (x + w, y + h), (200, 200, 200), 2)
            cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 200,), 2)
        
    cv2.imshow("Reconociendo en vivo optimizado", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(f"Se ha cerrado")
        break

cap.release()
cv2.destroyAllWindows()
                                    












