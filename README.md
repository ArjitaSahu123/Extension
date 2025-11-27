
#        AI vs REAL IMAGE CLASSIFIER – CHROME EXTENSION


A lightweight Chrome Extension integrated with a Flask backend
to classify whether an image is REAL or AI-GENERATED using a 
deep learning model (ResNet-50).

## 1. PROJECT OVERVIEW

This project detects if an image is REAL or AI-generated directly 
inside the browser. The Chrome Extension sends an image to a Flask 
server, which processes it using a trained neural network and 
returns a classification result with confidence score.

Technologies:
- Chrome Extension (HTML, CSS, JavaScript)
- Flask Backend (Python)
- ResNet-50 Classifier (PyTorch)
- DCGAN-generated fake images
- REST API communication

## 2. FEATURES

- Classifies Real vs AI-generated images
- Simple and clean browser UI
- Right-click "Check Image" option
- Supports image URLs and raw bytes
- Fast inference with Flask API
- Lightweight, modular, and upgrade-friendly

## 3. SYSTEM ARCHITECTURE

User → Chrome Extension → Flask Server → AI Model (ResNet50) → Result

Flow:
1. User clicks extension or right-clicks image
2. Extension extracts image source
3. Sends POST request to Flask backend
4. Backend preprocesses image (224x224, normalize)
5. ResNet-50 predicts REAL or FAKE
6. Backend returns JSON response
7. Extension displays the output


## 4. DATASET & MODEL 

Dataset Sources:
- Real Images: CelebA, CIP Real
- Fake Images: DCGAN-generated, CIP Fake

Preprocessing:
- Resize to 224x224
- Normalize pixel values
- Balanced dataset (equal real/fake)

Model:
- ResNet-50 (Transfer Learning)
- Modified final layer → binary output
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall, F1
 
## 5. INSTALLATION & SETUP
 
Backend Setup:
----------------
1. Install Python 3.8+
2. Install dependencies:
   pip install flask torch torchvision pillow requests

3. Run Flask server:
   python app.py

Chrome Extension Setup:
-----------------------
1. Open Chrome → Extensions → Developer Mode → Load Unpacked
2. Select the extension folder
3. Ensure Flask server is running
4. Click the extension icon and test any image

 
## 6. FILE STRUCTURE
 
project/
│
├── backend/
│   ├── app.py
│   ├── model/resnet50.pth
│   └── utils/preprocess.py
│
└── extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── content_script.js
    └── background.js

 
## 7. TOOLS & TECHNOLOGIES 
Frontend: HTML, CSS, JavaScript
Backend: Python Flask
Model Training: PyTorch, TorchVision
Datasets: CelebA, DCGAN generated faces
Browser API: Chrome Extensions (Manifest V3)
Version Control: Git & GitHub

Hardware Recommended:
- 8GB+ RAM
- GPU (optional) for faster training
 
## 8. HOW IT WORKS (SUMMARY)
 
- User selects an image
- Extension sends it to backend
- Backend preprocesses & runs model inference
- Model identifies REAL or AI-generated
- Result shown in popup window

## 9. RESULTS (SUMMARY)
 
- ResNet50 model trained and validated
- Stable performance on internal & external images
- Good generalization on real-world test cases

 
10. AUTHORS
 
- Arjita Sahu
- Aftab Alam
- Anam Fatima
- Abhishek Dwivedi
 
