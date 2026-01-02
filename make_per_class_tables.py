#!/usr/bin/env python3
"""
FAST per-class tables (Slice + Patient level)
GPU-optimized, fold-correct, binary classification
"""

import os, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda import amp

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report

# ---------------- ARGS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=0)  # keep 0 on Windows
parser.add_argument("--tta", action="store_true")
parser.add_argument("--folds", type=int, default=5)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 300
CLASSES = ["Non Demented", "Demented"]

print(f"Device: {DEVICE}")

# ---------------- TRANSFORMS ----------------
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- DATA ----------------
def get_pid(fname):
    return "_".join(fname.split("_")[:2])

def load_patient_map(root):
    mapping = {
        "Non Demented": 0,
        "Very mild Dementia": 1,
        "Demented": 1
    }
    patient_map = defaultdict(list)
    for cls, lbl in mapping.items():
        d = os.path.join(root, cls)
        for f in os.listdir(d):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                pid = get_pid(f)
                patient_map[pid].append((os.path.join(d,f), lbl))
    return patient_map

class SliceDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p,l,pid = self.items[idx]
        img = Image.open(p).convert("RGB")
        return val_tf(img), l, pid

# ---------------- MODEL (EXACT MATCH) ----------------
class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//16),
            nn.ReLU(),
            nn.Linear(ch//16, ch),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        w = self.fc(self.avg(x).view(b,c)).view(b,c,1,1)
        return x * w

class MLPMixerHead(nn.Module):
    def __init__(self, in_f, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f,512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,n)
        )
    def forward(self,x): return self.net(x)

def create_model():
    model = models.efficientnet_v2_s(weights=None)
    with torch.no_grad():
        feat = model.features(torch.zeros(1,3,IMG_SIZE,IMG_SIZE))
    ch = feat.shape[1]
    model.features.add_module("SE_Attention", SEBlock(ch))
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        MLPMixerHead(in_f, 2)
    )
    return model.to(DEVICE)

# ---------------- EVAL ----------------
def predict(model, loader):
    model.eval()
    slice_preds, slice_labels = [], []
    patient_probs = defaultdict(list)
    patient_true = {}

    soft = nn.Softmax(dim=1)

    with torch.no_grad():
        for x,y,pids in tqdm(loader, leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.numpy()

            with amp.autocast():
                if args.tta:
                    p1 = soft(model(x))
                    p2 = soft(model(torch.flip(x,[3])))
                    probs = ((p1+p2)/2).cpu().numpy()
                else:
                    probs = soft(model(x)).cpu().numpy()

            preds = probs.argmax(1)
            slice_preds += preds.tolist()
            slice_labels += y.tolist()

            for pid,l,pr in zip(pids,y,probs):
                patient_probs[pid].append(pr)
                patient_true[pid] = l

    patient_preds, patient_labels = [], []
    for pid, plist in patient_probs.items():
        avg = np.mean(plist, axis=0)
        patient_preds.append(avg.argmax())
        patient_labels.append(patient_true[pid])

    return (
        np.array(slice_labels), np.array(slice_preds),
        np.array(patient_labels), np.array(patient_preds)
    )

# ---------------- MAIN ----------------
patient_map = load_patient_map(args.data)
patient_ids = sorted(patient_map.keys())
patient_labels = np.array([max(set(l for _,l in patient_map[p]), key=[l for _,l in patient_map[p]].count) for p in patient_ids])

gkf = GroupKFold(n_splits=args.folds)

slice_reports = []
patient_reports = []

for fold,(tr,te) in enumerate(gkf.split(patient_ids, patient_labels, groups=patient_ids)):
    print(f"\nEvaluating fold {fold}")

    test_pids = [patient_ids[i] for i in te]

    test_items = []
    for pid in test_pids:
        for p,l in patient_map[pid]:
            test_items.append((p,l,pid))

    loader = DataLoader(
        SliceDataset(test_items),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = create_model()
    ckpt = torch.load(os.path.join(args.out, f"fold_{fold}", "best_stage2.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    sl, sp, pl, pp = predict(model, loader)

    slice_reports.append(classification_report(sl, sp, output_dict=True, zero_division=0))
    patient_reports.append(classification_report(pl, pp, output_dict=True, zero_division=0))

# ---------------- PRINT TABLES ----------------
def print_table(reports, title):
    print(f"\n{title}")
    print("-"*60)
    print(f"{'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-"*60)
    for i,c in enumerate(CLASSES):
        p = np.mean([r[str(i)]['precision'] for r in reports])
        r = np.mean([r[str(i)]['recall'] for r in reports])
        f = np.mean([r[str(i)]['f1-score'] for r in reports])
        print(f"{c:15s} {p:10.3f} {r:10.3f} {f:10.3f}")
    mp = np.mean([r['macro avg']['precision'] for r in reports])
    mr = np.mean([r['macro avg']['recall'] for r in reports])
    mf = np.mean([r['macro avg']['f1-score'] for r in reports])
    print("-"*60)
    print(f"{'Macro Avg.':15s} {mp:10.3f} {mr:10.3f} {mf:10.3f}")

print_table(slice_reports, "TABLE: Slice-Level Per-Class Metrics")
print_table(patient_reports, "TABLE: Patient-Level Per-Class Metrics")
