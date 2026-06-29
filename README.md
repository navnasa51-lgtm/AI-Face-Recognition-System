# AI-Face-Recognition-System


An optimized, real-time edge AI surveillance system built using Python. This project combines **YOLOv8** for seamless object and person tracking with **InsightFace (ArcFace)** for spatial multi-person facial recognition triggered at smart intervals.

## 🚀 Key Features
- **Persistent Bounding Boxes**: Object tracking runs frame-by-frame smoothly without stuttering or skipping.
- **Interval-Based Face Recognition**: Heavy ArcFace facial embedding checks are computed every 5 seconds to minimize CPU load while preserving system responsiveness.
- **Multi-Person Spatial Coordinate Mapping**: Dynamically binds face identities to unique individual YOLO bounding boxes, preventing cross-identity mapping errors.
- **Strict Matching Threshold**: Configured to separate family members/lookalikes accurately and correctly tag un-registered profiles as `Unknown`.

---

## 📁 Repository Structure
* `main.py`: The main surveillance execution loop script handling video feed streams, object tracking, and identity mapping.
* `train_faces.py`: Script used to encode a new user's face and append it into the embeddings database.
* `database/` : Directory where the images used to train are kept. *(Note: This folder can be created on your own manually, or it will generate automatically when running `train_faces.py`)*.
* `embeddings/`: Directory where the local face feature data matrix (`face_db.npy`) is kept. *(Note: This folder will generate automatically when running `train_faces.py`)*.

---

## 🛠️ Prerequisites & Installation

### 1. Clone the repository ###
```bash
git clone [https://github.com/muhammedafnasok-design/ai-surveillance-system.git](https://github.com/muhammedafnasok-design/ai-surveillance-system.git)
cd ai-surveillance-system
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
[!NOTE]

The database/ and embeddings/ folders will automatically generate when you execute the script below. Alternatively, you can create the database/ folder by yourself beforehand if you want to place your target user image folders inside manually.
### 3. Train Faces
```
python train_faces.py
```
### 4. Run the Project
```
python main.py
```


