#!/usr/bin/env python3
"""
app.py - Flask application for Alzheimer's MRI Classification
FIXED VERSION - Model structure matches saved weights
"""

import os
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, request, jsonify

# ==================== CONFIGURATION ====================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./cv_results_binary"
NUM_FOLDS = 5
USE_ENSEMBLE = True

# ==================== FLASK SETUP ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== MODEL COMPONENTS ====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(4, channels//reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels//reduction), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class MLPMixerHead(nn.Module):
    """MLP Mixer classification head"""
    def __init__(self, in_dim, num_classes, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x): 
        return self.net(x)

# ==================== MODEL CREATION ====================
def build_model():
    """
    Build model architecture that matches saved weights.
    Structure: EfficientNetV2-S + SE Block + Dropout + MLPMixer
    NO GeM pooling (use_gem=False)
    """
    # Base model
    model = models.efficientnet_v2_s(weights=None)
    
    # Probe to get feature dimensions
    model.eval()
    with torch.no_grad():
        try:
            feat = model.features(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        except:
            feat = model.features(torch.zeros(1, 3, 224, 224))
    
    channels = feat.shape[1]
    
    # Add SE attention block to features
    model.features.add_module("SE_Attention", SEBlock(channels))
    
    # Get classifier input dimension
    try:
        in_features = model.classifier[1].in_features
    except:
        in_features = 1280
    
    # Build classifier WITHOUT GeM
    # This creates: classifier[0]=Dropout, classifier[1]=MLPMixer
    # Which matches the saved weights: classifier.1.net.X
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),                      # index 0
        MLPMixerHead(in_features, 2, 0.3)    # index 1
    )
    
    return model

# ==================== MODEL LOADING ====================
def load_all_models():
    """Load trained model checkpoints"""
    loaded_models = []
    
    if USE_ENSEMBLE:
        print(f"\nLoading ensemble of {NUM_FOLDS} models...")
        for fold in range(NUM_FOLDS):
            model_path = os.path.join(MODEL_DIR, f"fold_{fold}", "best_stage2.pth")
            
            if not os.path.exists(model_path):
                print(f"  Fold {fold}: Model not found at {model_path}")
                continue
            
            try:
                model = build_model()
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(DEVICE)
                model.eval()
                loaded_models.append(model)
                print(f"  Fold {fold}: ✓ Loaded successfully")
                
            except Exception as e:
                print(f"  Fold {fold}: ✗ Error loading - {e}")
    
    else:
        # Single model mode
        model_path = os.path.join(MODEL_DIR, "fold_0", "best_stage2.pth")
        print(f"\nLoading single model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = build_model()
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(DEVICE)
        model.eval()
        loaded_models.append(model)
        print("✓ Model loaded successfully")
    
    if not loaded_models:
        raise RuntimeError("No models could be loaded!")
    
    return loaded_models

# ==================== IMAGE PREPROCESSING ====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Convert image bytes to preprocessed tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# ==================== INFERENCE ====================
def predict(image_tensor, models):
    """Run inference on image"""
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        if len(models) > 1:
            # Ensemble prediction - average probabilities
            all_probs = []
            for model in models:
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)
            
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred_class = torch.argmax(avg_probs, dim=1).item()
            confidence = avg_probs[0][pred_class].item()
            
        else:
            # Single model prediction
            output = models[0](image_tensor)
            avg_probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(avg_probs, dim=1).item()
            confidence = avg_probs[0][pred_class].item()
    
    class_names = ["Non Demented", "Demented"]
    
    return {
        "prediction": int(pred_class),
        "class_name": class_names[pred_class],
        "confidence": float(confidence),
        "probabilities": {
            "Non Demented": float(avg_probs[0][0].item()),
            "Demented": float(avg_probs[0][1].item())
        }
    }

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handle prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        result = predict(image_tensor, MODELS)
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'device': DEVICE,
        'ensemble_mode': USE_ENSEMBLE
    })

# ==================== STARTUP ====================
print("\n" + "="*70)
print("ALZHEIMER'S MRI CLASSIFICATION SERVER")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Model Directory: {MODEL_DIR}")
print(f"Ensemble Mode: {USE_ENSEMBLE}")

try:
    MODELS = load_all_models()
    print(f"\n{'='*70}")
    print(f"✓ READY - {len(MODELS)} model(s) loaded successfully")
    print(f"{'='*70}\n")
except Exception as e:
    print(f"\n{'='*70}")
    print(f"✗ STARTUP FAILED")
    print(f"Error: {e}")
    print(f"{'='*70}\n")
    MODELS = []

# ==================== MAIN ====================
if __name__ == '__main__':
    if not MODELS:
        print("WARNING: No models loaded - predictions will fail!")
    
    print("Starting server at http://localhost:5000")
    print("Press CTRL+C to quit\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)