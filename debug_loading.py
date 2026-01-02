#!/usr/bin/env python3
"""
debug_loading.py - Debug script to test model loading
"""

import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 300

class SEBlock(nn.Module):
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

def create_model_without_gem():
    """Create model WITHOUT GeM - matches your saved weights"""
    model = models.efficientnet_v2_s(weights=None)
    
    # Add SE block
    model.eval()
    with torch.no_grad():
        try:
            feat = model.features(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        except:
            feat = model.features(torch.zeros(1, 3, 224, 224))
    channels = feat.shape[1]
    model.features.add_module("SE_Attention", SEBlock(channels))
    
    # Get input features
    try:
        in_f = model.classifier[1].in_features
    except:
        in_f = 1280
    
    # Structure WITHOUT GeM (matches classifier.1.net.X)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),                           # index 0
        MLPMixerHead(in_f, 2, drop=0.3)           # index 1
    )
    
    return model

# Test loading
print("="*60)
print("Testing Model Loading")
print("="*60)

model_path = "./cv_results_binary/fold_0/best_stage2.pth"
print(f"\nLoading from: {model_path}")

try:
    model = create_model_without_gem()
    print("\n✓ Model architecture created")
    
    print("\nModel classifier structure:")
    for i, layer in enumerate(model.classifier):
        print(f"  [{i}] {layer.__class__.__name__}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    print("\nClassifier keys in saved model:")
    classifier_keys = [k for k in state_dict.keys() if k.startswith('classifier')]
    for key in sorted(classifier_keys)[:10]:
        print(f"  {key}")
    
    print("\nAttempting to load state_dict...")
    model.load_state_dict(state_dict)
    print("✓ SUCCESS! Model loaded without errors")
    
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nThis means the model structure still doesn't match!")