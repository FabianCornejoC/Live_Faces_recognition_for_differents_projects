from deepface import DeepFace
import os
import numpy as np

# image rut
dataset_path = input("Ruta del data set (Ej. C:\A\B): ").strip()

# save embeds
embeddings_name = input("Dirección del hogar (Ej. Alberto80Rancagua): ").strip()

# File for save embeddings

save_dir = input("Ruta para guardar embeddings (Ej. C:\A\B\C): " ).strip() # ruta especifica
os.makedirs(save_dir, exist_ok=True) # Create file
embeddings_file = os.path.join(save_dir, embeddings_name + '.npy')

# dictonary for embeddings

embeddings_dic = {}

# Run person files

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings_list = []

    for img_name in os.listdir(person_dir):
        if img_name.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(person_dir, img_name)
            try:
                embedding = DeepFace.represent(  # take picture and create his vectors (embeddings), represent the face. 
                    img_path = img_path, 
                    model_name = "Facenet", 
                    enforce_detection = True
                )                                 
                
                embedding_vector = np.array(embedding[0]["embedding"]).flatten()
                embeddings_list.append(embedding_vector)

            except Exception as e:
                print(f"No se pudo procesar {img_path}: {e}")


    # saving all person embeddings
    if embeddings_list:
        embeddings_dic[person_name] = embeddings_list # embeddigs of each person
        print(f"Procesados {len(embeddings_list)} embeddings para {person_name}")
    else:
        print(f"No se pudo generar embeddings para {person_name}")
        


# Saving file for use

np.save(embeddings_file, embeddings_dic) # save all embeddings in one file
print(f"Embeddings guardados en {embeddings_file}")



