import cv2
import numpy as np
import time

from ultralytics import YOLO
from insightface.app import FaceAnalysis

# =====================================================
# OPENCV OPTIMIZATION
# =====================================================

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# =====================================================
# SETTINGS
# =====================================================

YOLO_SIZE = 640
FACE_RECOGNITION_INTERVAL = 5.0  # Check faces every 5 seconds
FACE_THRESHOLD = 0.50  # Raised to 0.50 to stop false positives (like your brother matching you)
MIN_PERSON_SIZE = 80

# =====================================================
# LOAD YOLO (ONNX RECOMMENDED)
# =====================================================

print("Loading YOLO...")
yolo = YOLO("yolov8n.pt")
print("YOLO Loaded")

# =====================================================
# LOAD INSIGHTFACE
# =====================================================

print("Loading ArcFace...")
face_app = FaceAnalysis()
face_app.prepare(
    ctx_id=-1,  # CPU
    det_size=(640, 480)
)
print("ArcFace Loaded")

# =====================================================
# LOAD FACE DATABASE
# =====================================================

print("Loading Face Database...")
database = np.load(
    "embeddings/face_db.npy",
    allow_pickle=True
)

known_names = []
known_embeddings = []

for person in database:
    known_names.append(person["name"])
    known_embeddings.append(person["embedding"])

known_embeddings = np.array(
    known_embeddings,
    dtype=np.float32
)

print(f"Loaded {len(known_names)} embeddings")

# =====================================================
# CLASSES OF INTEREST
# =====================================================

INTEREST_CLASSES = [
    "person", "cell phone", "laptop", "bottle",
    "backpack", "chair", "book", "mouse", "keyboard"
    "umbrella", "bed", "cup", "fork", "spoon", "knife"
]

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot access camera")
    exit()

# =====================================================
# PERFORMANCE & MULTI-TRACKING VARIABLES
# =====================================================

# Dictionary to hold face embeddings mapped directly to coordinates to prevent sharing names
current_frame_identities = {}
last_face_timestamp = 0.0

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    # =================================================
    # YOLO DETECTION
    # =================================================

    results = yolo(
        frame,
        imgsz=YOLO_SIZE,
        verbose=False
    )

    boxes = results[0].boxes
    current_time = time.time()

    # =================================================
    # MULTI-PERSON FACE RECOGNITION STEP (EVERY 5 SECS)
    # =================================================
    if current_time - last_face_timestamp >= FACE_RECOGNITION_INTERVAL:
        # Reset current frame identities map on every cycle run
        current_frame_identities.clear()

        try:
            faces = face_app.get(frame)

            # Map faces detected in this interval to their absolute spatial position indices
            for idx, face in enumerate(faces):
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                embedding = face.embedding

                # Check similarity against database
                similarities = (known_embeddings @ embedding).flatten()
                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])

                print(f"[Face Check] Matched index {idx} to {known_names[best_idx]} with score {best_score:.4f}")

                # Assign name safely based on strict verification limits
                if best_score >= FACE_THRESHOLD:
                    name = known_names[best_idx]
                else:
                    name = "Unknown"

                # Save data with coordinates as the key reference point
                current_frame_identities[idx] = {
                    "bbox": (fx1, fy1, fx2, fy2),
                    "name": name
                }

            last_face_timestamp = current_time

        except Exception as e:
            print(f"Recognition Error: {e}")
            pass

    # =================================================
    # PROCESS DETECTIONS AND DRAW
    # =================================================

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = yolo.names[class_id]

        if label not in INTEREST_CLASSES:
            continue

        # =============================================
        # PERSON PROCESSING
        # =============================================
        if label == "person":
            width = x2 - x1
            height = y2 - y1

            if width < MIN_PERSON_SIZE or height < MIN_PERSON_SIZE:
                continue

            # Default identity for this specific frame detection block
            assigned_name = "Unknown"

            # Find which face belongs structurally inside this specific person bounding box
            for face_idx, data in current_frame_identities.items():
                fx1, fy1, fx2, fy2 = data["bbox"]

                # Spatial checking logic with 30px padding buffer allowance
                if (fx1 >= x1 - 30 and fx2 <= x2 + 30 and
                        fy1 >= y1 - 30 and fy2 <= y2 + 30):
                    assigned_name = data["name"]
                    break

            # Draw visual borders for individual person
            color = (0, 255, 0) if assigned_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                assigned_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # =============================================
        # OTHER OBJECTS
        # =============================================
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

    # =================================================
    # DISPLAY
    # =================================================
    cv2.imshow("AI Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================================
# CLEANUP
# =====================================================
cap.release()
cv2.destroyAllWindows()