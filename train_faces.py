import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from torch.nn.functional import embedding

app = FaceAnalysis()
app.prepare(ctx_id=0)

database = []
root = "database"

for person in os.listdir(root):
    person_path = os.path.join(root,person)
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path,image_name)
        img = cv2.imread(image_path)
        faces = app.get(img)
        if len(faces) == 0:
            continue

        embedding = faces[0].embedding
        database.append({"name": person, "embedding": embedding})
np.save("embeddings/face_db", database)
print("Training Completed")