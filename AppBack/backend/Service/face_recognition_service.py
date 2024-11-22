# FACECAMERA/Service/face_recognition_service.py

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import jwt
from datetime import datetime, timedelta
from functools import wraps
from keras_facenet import FaceNet
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import django
from sklearn.preprocessing import Normalizer
import base64
from Service.config import Config
import jwt
# Akses SECRET_KEY dari kelas Config
SECRET_KEY = Config.SECRET_KEY

# Atur DJANGO_SETTINGS_MODULE ke jalur pengaturan Anda
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from app.face_recognition_training import FaceRecognitionTraining
from app.models import Employee

app = Flask(__name__)
CORS(app)

# Konfigurasi
app.config.from_object('Service.config.Config')

# Inisialisasi model dan detector
embedder = FaceNet()
detector = MTCNN()
trainer = FaceRecognitionTraining()
norm_encoder = Normalizer(norm='l2')

# Load model dan labels
model = load_model(app.config['MODEL_PATH'])
with open(app.config['LABELS_PATH'], 'r') as file:
    LABELS = [line.strip() for line in file.readlines()]

# Liveness detection cascades
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables for tracking
prev_bbox = None
movement_threshold = 10

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(" ")[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = Employee.objects.get(id=data['user_id'])
        except Exception:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def detect_smile_and_eyes(face_img):
    gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    smiles = smile_cascade.detectMultiScale(gray_face_img, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    smile_detected = len(smiles) > 0

    eyes = eye_cascade.detectMultiScale(gray_face_img, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
    eye_detected = len(eyes) >= 2

    return smile_detected, eye_detected

def is_head_moving(current_bbox, prev_bbox, threshold):
    if prev_bbox is None:
        return False
    x1, y1, w1, h1 = current_bbox
    x2, y2, w2, h2 = prev_bbox
    movement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return movement > threshold

def predict_identity(embedding, threshold=0.9):
    embedding_norm = norm_encoder.transform(embedding.reshape(1, -1))
    predictions = model.predict(embedding_norm)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    if confidence < threshold:
        return None, 0
    
    predicted_label = LABELS[predicted_class]
    try:
        employee = Employee.objects.get(name=predicted_label.split()[0])
        return employee, confidence * 100
    except Employee.DoesNotExist:
        return None, 0

def process_frame(frame_data):
    global prev_bbox
    
    try:
        # Decode base64 image
        if isinstance(frame_data, str) and frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Convert base64 to numpy array
        img_data = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, "Invalid frame data"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        if not faces:
            return None, "No face detected"

        face = faces[0]
        bbox = face['box']
        x, y, w, h = bbox
        face_img = rgb_frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))

        head_moved = is_head_moving(bbox, prev_bbox, movement_threshold)
        prev_bbox = bbox

        smile_detected, eyes_detected = detect_smile_and_eyes(face_img)

        if not (smile_detected and eyes_detected and head_moved):
            return None, "Liveness check failed"

        embedding = trainer.get_embedding(face_img)
        identity, confidence = predict_identity(embedding)

        if not identity:
            return None, "Identity not recognized"

        return identity, confidence

    except Exception as e:
        return None, str(e)

# API Endpoints
@app.route('/api/face-login', methods=['POST'])
def face_login():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({'message': 'Missing JSON in request'}), 400

        frame_data = request.json.get('frame')
        if not frame_data:
            return jsonify({'message': 'No frame data provided'}), 400

        identity, message = process_frame(frame_data)
        
        if identity is None:
            return jsonify({
                'success': False,
                'message': message
            }), 400

        # Create JWT token
        token = jwt.encode({
            'user_id': identity.id,
            'username': identity.username,
            'role': identity.role,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': identity.id,
                'username': identity.username,
                'role': identity.role,
                'email': identity.email
            },
            'redirect_url': f"http://127.0.0.1:5000/success"
        })
    else:
        return jsonify({'message': 'Method Not Allowed'}), 405

@app.route('/api/verify-token', methods=['GET'])
@token_required
def verify_token(current_user):
    return jsonify({
        'valid': True,
        'user': {
            'username': current_user.username,
            'role': current_user.role
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
