#!/usr/bin/env python3
"""
inspect_model.py - Diagnostic script to see what's in your saved model
Run with: python inspect_model.py
"""

import torch
import os

MODEL_DIR = "./cv_results_binary"
model_path = os.path.join(MODEL_DIR, "fold_0", "best_stage2.pth")

print(f"Loading model from: {model_path}")
print("=" * 60)

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Extract state dict
if isinstance(checkpoint, dict):
    print("Checkpoint is a dictionary with keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Using 'model_state_dict' from checkpoint\n")
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint
    print("Checkpoint is directly a state_dict\n")

print("=" * 60)
print("CLASSIFIER LAYERS IN SAVED MODEL:")
print("=" * 60)

classifier_keys = [key for key in state_dict.keys() if 'classifier' in key]

for key in sorted(classifier_keys):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"{key:50s} -> {shape}")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)

# Analyze the structure
has_gem = any('classifier.0.p' in key for key in classifier_keys)
has_dropout_at_2 = any('classifier.2.' in key for key in classifier_keys)
has_mlp_at_1 = any('classifier.1.net' in key for key in classifier_keys)
has_mlp_at_3 = any('classifier.3.net' in key for key in classifier_keys)

print(f"Has GeM at index 0 (classifier.0.p): {has_gem}")
print(f"Has MLPMixer at index 1 (classifier.1.net): {has_mlp_at_1}")
print(f"Has Dropout at index 2: {has_dropout_at_2}")
print(f"Has MLPMixer at index 3 (classifier.3.net): {has_mlp_at_3}")

print("\n" + "=" * 60)
print("RECOMMENDED MODEL STRUCTURE:")
print("=" * 60)

if has_mlp_at_1 and not has_gem:
    print("""
Your model was trained WITHOUT GeM pooling!
    
Use this in create_model():
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),                                # index 0
        MLPMixerHead(in_f, num_classes, drop=0.3)      # index 1
    )
""")
elif has_mlp_at_3 and has_gem:
    print("""
Your model was trained WITH GeM pooling!
    
Use this in create_model():
    model.classifier = nn.Sequential(
        GeM(),                                          # index 0
        nn.Flatten(),                                   # index 1
        nn.Dropout(0.4),                                # index 2
        MLPMixerHead(in_f, num_classes, drop=0.3)      # index 3
    )
""")
else:
    print("Unusual structure detected. Check the keys above manually.")

print("=" * 60)