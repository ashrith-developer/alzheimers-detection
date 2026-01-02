#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda import amp

from compute_slice_level_metrics import (
    build_patient_index,
    SliceDataset,
    create_model,
)

# ---------------- ARGS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./Data")
parser.add_argument("--out", default="./cv_results_binary")
parser.add_argument("--binary", action="store_true")
parser.add_argument("--img_size", type=int, default=300)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--fold", type=int, default=0, help="Fold index")
parser.add_argument("--tta", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ----------------
val_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- LOAD DATA ----------------
classes, _, patient_map, patient_label = build_patient_index(args.data, binary=args.binary)

items = []
for pid, lst in patient_map.items():
    for p,l in lst:
        items.append((p,l,pid))

ds = SliceDataset(items, transform=val_tf)
loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers)

# ---------------- LOAD MODEL ----------------
model = create_model(len(classes))
ckpt = torch.load(
    os.path.join(args.out, f"fold_{args.fold}", "best_stage2.pth"),
    map_location=DEVICE
)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.to(DEVICE)
model.eval()

soft = nn.Softmax(dim=1)

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        with amp.autocast():
            out = model(imgs)
            probs = soft(out)
        preds = probs.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", values_format="d")
plt.title("Slice-level Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_slice.png", dpi=300)
