# FaceCamera
## Introduction
FaceCamera is a face recognition system designed to facilitate real-time identity verification for logging into a webpage based on employee status. Using a combination of Cascade Classifier for face detection, FaceNet for feature extraction, and Artificial Neural Network (ANN) for classification, the system provides accurate and efficient face recognition capabilities. As well as detecting using smiles and head movements. The language used is python with django admin database.
## Meet The Team
| NIM      | Name                         | University                             | Scope of Task                                                                                                                                                                                                                              |
| ---------| --------------------------   | ---------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 17210809 | Roslina Puspita              | Universitas Bina Sarana Informatika    | Model Research, Dataset Collection, Data Processing, Deep Learning Model Development, Model Training, Testing & Optimization, Model Deployment, Real-Time Face Recognition, Model Training, ML-Frontend Integration, ML Backend, Frontend & Backend initial view, ML for Smile and Head Movement, Camera, Documentation |
| 17210640 | Lailatul Qodariyah           | Universitas Bina Sarana Informatika    | Model Research, Dataset Collection, Refine the initial look of the Backend, Backend creation of enhancements that have been made |
| 19210759 | Syifa Rahma Leily            | Universitas Bina Sarana Informatika    | Part of the Backend, Create a login on the backend that only supervisors can enter | |
| 19210782 | Fransisca Kusuma             | Universitas Bina Sarana Informatika    | Creating Frontend Views using Frames and BI Logos |
| 15210380 | Raihan Juniargho             | Universitas Bina Sarana Informatika    | Creating Frontend Views using Frames and BI Logos |

## Flow Diagram for Face Detection and Recognition
The following diagram illustrates the step-by-step process for face detection and recognition using MTCNN to detect faces, FaceNet to extract facial features, and ANN to classify faces as well as detection using smiles and head movements :

<img width="332" alt="Screenshot 2024-11-07 151556" src="https://github.com/user-attachments/assets/65f96918-6217-43dc-bec9-c6c5736af80b">


## Requirements
To run this project, you need the following Python packages with their specified versions :
- absl-py==2.1.0
- asgiref==3.8.1
- astunparse==1.6.3
- blinker==1.8.2
- certifi==2024.8.30
- charset-normalizer==3.4.0
- click==8.1.7
- colorama==0.4.6
- Django==5.1.2
- Flask==3.0.3
- flatbuffers==24.3.25
- gast==0.6.0
- google-pasta==0.2.0
- grpcio==1.67.0
- h5py==3.12.1
- idna==3.10
- itsdangerous==2.2.0
- Jinja2==3.1.4
- joblib==1.4.2
- keras==3.6.0
- keras-facenet==0.3.2
- libclang==18.1.1
- lz4==4.3.3
- Markdown==3.7
- markdown-it-py==3.0.0
- MarkupSafe==3.0.2
- mdurl==0.1.2
- ml-dtypes==0.4.1
- mtcnn==1.0.0
- namex==0.0.8
- numpy==2.0.2
- opencv-python==4.10.0.84
- opt_einsum==3.4.0
- optree==0.13.0
- packaging==24.1
- pillow==11.0.0
- protobuf==5.28.3
- Pygments==2.18.0
- requests==2.32.3
- rich==13.9.3
- scikit-learn==1.5.2
- scipy==1.14.1
- six==1.16.0
- sqlparse==0.5.1
- tensorboard==2.18.0
- tensorboard-data-server==0.7.2
- tensorflow==2.18.0
- tensorflow-io-gcs-filesystem==0.31.0
- tensorflow_intel==2.18.0
- termcolor==2.5.0
- threadpoolctl==3.5.0
- typing_extensions==4.12.2
- tzdata==2024.2
- urllib3==2.2.3
- Werkzeug==3.0.5
- wrapt==1.16.0

## This Project Allow :
- Python 3.10.5

Check the following installation if there is no please install this :
- flask 
- cv2
- face_recognition
- numpy

## Installation
To set up the project locally, follow these steps:
(Create a new folder with the name FaceCamera and clone the project in this directory)

1. Clone the repository:
   ```bash
   git clone https://github.com/BI-Presence/FaceCamera.git
   cd FaceCamera
    ```
   
2. Create a virtual environment:
   ```bash
   python -m venv my_env
   ```

3. Activate the virtual environment:
   ```bash
   my_env\Scripts\activate
    ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install other package in description :
    - flask
   - cv2
   - face_recognition
   - numpy

6. Run Camera:
    ```bash
    cd FrontEnd
    python camera.py
    ```

7. Access the Camera:
   Open your web browser and go to http://127.0.0.1:8001
   
8. Run Backend For Supervisor:
    ```bash
    cd AppBack
    cd backend
    python manage.py runserver
    ```
    
9. Access the Backend:
   Open your web browser and go to http://127.0.0.1:8000


Folder:
- FrontEnd folder also the Frontend Camera and templates camera in this Folder.
- AppBack folder is a folder that contains the Backend in the project created both ML and also the Backend View

Contact me if you have any questions about the project at puspitaroslina@gmail.com
