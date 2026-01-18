from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
import uuid

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATABASE = 'emotions.db'
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face detector 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load the NEW Model ---
# This MUST match the name in model.py
MODEL = tf.keras.models.load_model("emotion_model_v2.h5")

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize DB
with get_db_connection() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, image_path TEXT, emotion TEXT
        )
    """)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        image_file = request.files['image']

        unique_filename = f"{uuid.uuid4().hex}_{image_file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        image_file.save(image_path)

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # 1. Crop original COLOR image
            face_roi = img[y:y+h, x:x+w]
            
            # 2. Resize to 224x224 (The new standard)
            final_img = cv2.resize(face_roi, (224, 224))
            
            # 3. Convert to RGB for MobileNet
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            
            # 4. Normalize & Predict
            final_img = final_img / 255.0
            final_img = np.expand_dims(final_img, axis=0)
            
            predictions = MODEL.predict(final_img)
            emotion = LABELS[np.argmax(predictions)]
        else:
            emotion = "No Face Detected"

        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (name, image_path, emotion) VALUES (?, ?, ?)",
                         (name, image_path, emotion))
            conn.commit()

        display_path = image_path.replace("\\", "/")
        return render_template('index.html', result=emotion, img=display_path)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)