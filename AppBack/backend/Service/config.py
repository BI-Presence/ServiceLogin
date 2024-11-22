# FACECAMERA/Service/config.py

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', '3c2e719060076f0bdd324dfe15f4111bd88266cb573f1ddb')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             '../backend/media/mtcnn_facenet_ann_model.h5')
    LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              '../backend/media/face_labels.txt')
    
    # Database configuration (menggunakan database yang sudah ada)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///db.sqlite3'
    SQLALCHEMY_TRACK_MODIFICATIONS = False