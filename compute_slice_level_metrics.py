#!/usr/bin/env python3
"""
Compute slice-level Macro Precision / Recall / F1 (per-fold -> mean ± std).
Designed to match your training GroupKFold splits and model layout.

Usage example:
    python compute_slice_level_metrics.py --data ./Data --out ./cv_results_binary --folds 5 --img_size 300 --batch 64 --binary --tta --num_workers 0
"""

import os
import argparse
from collections import defaultdict, Counter
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda import amp

# ----------------- ARGS -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./Data", help="Root folder with class subfolders")
parser.add_argument("--out", default="./cv_results_binary", help="Folder with fold_x subfolders containing checkpoints")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--img_size", type=int, default=300)
parser.add_argument("--tta", action="store_true", help="Use simple TTA (hflip)")
parser.add_argument("--binary", action="store_true", help="Merge Very mild + Demented into Demented")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")
print(f"Using folds={args.folds}, batch={args.batch}, img_size={args.img_size}, TTA={args.tta}, binary={args.binary}")

# ----------------- TRANSFORMS -----------------
val_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------- DATA HELPERS -----------------
def get_patient_id(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else os.path.splitext(base)[0]

def build_patient_index(root, binary=False):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if binary:
        mapping = {"Non Demented": 0, "Very mild Dementia": 1, "Demented": 1}
        cls2idx = {}
        for c in classes:
            cls2idx[c] = mapping.get(c, max(mapping.values())+1)  # unknown -> appended
        final_classes = ["Non Demented", "Demented"]
    else:
        cls2idx = {c:i for i,c in enumerate(classes)}
        final_classes = classes

    patient_map = defaultdict(list)
    for c in classes:
        pdir = os.path.join(root, c)
        for f in os.listdir(pdir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            pid = get_patient_id(f)
            patient_map[pid].append((os.path.join(pdir, f), cls2idx[c]))

    patient_label = {}
    for pid, items in patient_map.items():
        lbls = [l for _, l in items]
        patient_label[pid] = Counter(lbls).most_common(1)[0][0]

    return final_classes, cls2idx, patient_map, patient_label

class SliceDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p,l,pid = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, l, pid

# ----------------- MODEL BUILD (match training) -----------------
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
    def forward(self,x):
        b,c,_,_ = x.size()
        w = self.avg(x).view(b,c)
        w = self.fc(w).view(b,c,1,1)
        return x * w

class MLPMixerHead(nn.Module):
    def __init__(self, in_dim, num_classes, drop=0.3):
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
    def forward(self,x): return self.net(x)

def create_model(num_classes:int, img_size=args.img_size):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.eval()
    with torch.no_grad():
        try:
            feat = model.features(torch.zeros(1,3,img_size,img_size))
        except Exception:
            feat = model.features(torch.zeros(1,3,224,224))
    channels = feat.shape[1]
    model.features.add_module("SE_Attention", SEBlock(channels))
    try:
        in_f = model.classifier[1].in_features
    except Exception:
        in_f = 1280
    model.classifier = nn.Sequential(nn.Dropout(0.4), MLPMixerHead(in_f, num_classes, drop=0.3))
    return model.to(DEVICE)

# ----------------- EVAL helpers -----------------
def tta_predict(model, imgs):
    views = [imgs, torch.flip(imgs, dims=[3])]
    soft = nn.Softmax(dim=1)
    probs = None
    with torch.no_grad():
        for v in views:
            v = v.to(DEVICE)
            with amp.autocast():
                out = model(v)
                p = soft(out).cpu().numpy()
            probs = p if probs is None else probs + p
    return probs / len(views)

def eval_slice_preds(model, loader, use_tta=False):
    model.eval()
    soft = nn.Softmax(dim=1)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Eval(slices)", leave=False):
            imgs = imgs.to(DEVICE)
            if use_tta:
                probs = tta_predict(model, imgs)
            else:
                with amp.autocast():
                    out = model(imgs)
                    probs = soft(out).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)

# ----------------- load checkpoint robustly -----------------
def load_model_from_ckpt(path, num_classes):
    ck = torch.load(path, map_location=DEVICE)
    # choose model architecture
    model = create_model(num_classes)
    # state could be dict with "model_state_dict" or direct
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck
    # quick compatibility: if classifier output dim in state mismatches, try to detect the final linear key and skip strict
    out_dim = None
    for k,v in state.items():
        if k.endswith("net.6.weight") or k.endswith("classifier.1.net.6.weight") or k.endswith("classifier.1.weight"):
            try:
                out_dim = v.shape[0]; break
            except Exception:
                pass
    if out_dim is not None and out_dim != num_classes:
        raise RuntimeError(f"Checkpoint at {path} has classifier output dim {out_dim} but num_classes={num_classes}. Use correct --binary flag or match out folder.")
    # load
    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint via strict load.")
    except Exception as e:
        print("Strict load failed, attempting non-strict load:", e)
        model.load_state_dict(state, strict=False)
    return model

# ----------------- MAIN -----------------
def main():
    classes, cls2idx, patient_map, patient_label = build_patient_index(args.data, binary=args.binary)
    num_classes = len(classes)
    print("Final classes:", classes)

    patient_ids = sorted(patient_map.keys())
    labels = np.array([patient_label[pid] for pid in patient_ids])
    pid2idx = {pid:i for i,pid in enumerate(patient_ids)}
    group_idxs = np.array([pid2idx[pid] for pid in patient_ids])

    gkf = GroupKFold(n_splits=args.folds)

    slice_precisions = []
    slice_recalls = []
    slice_f1s = []
    slice_accs = []

    for fold_idx, (tv_idx, test_idx) in enumerate(gkf.split(patient_ids, labels, groups=group_idxs)):
        print("\n========== Fold", fold_idx, "==========")
        test_pids = [patient_ids[i] for i in test_idx]

        test_items = []
        for pid in test_pids:
            for path,lbl in patient_map[pid]:
                test_items.append((path, lbl, pid))
        test_ds = SliceDataset(test_items, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(DEVICE=="cuda"))

        fold_dir = os.path.join(args.out, f"fold_{fold_idx}")
        cand = None
        for name in ("best_stage2.pth","checkpoint.pth","best_stage1.pth"):
            p = os.path.join(fold_dir, name)
            if os.path.exists(p):
                cand = p; break
        if cand is None:
            print(f"[WARN] No checkpoint found in {fold_dir}, skipping fold.")
            continue
        print("Fold", fold_idx, "loading", cand)
        model = load_model_from_ckpt(cand, num_classes)
        model.to(DEVICE); model.eval()

        y_true, y_pred = eval_slice_preds(model, test_loader, use_tta=args.tta)

        # per-fold metrics
        p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        print(f"Slice Macro -> P={p:.3f} R={r:.3f} F1={f:.3f} Acc={acc:.3f}")

        slice_precisions.append(p)
        slice_recalls.append(r)
        slice_f1s.append(f)
        slice_accs.append(acc)

    # summarize
    def mean_std(vals):
        vals = np.array(vals)
        return 100.0 * vals.mean(), 100.0 * vals.std()

    Pm, Ps = mean_std(slice_precisions) if slice_precisions else (np.nan,np.nan)
    Rm, Rs = mean_std(slice_recalls) if slice_recalls else (np.nan,np.nan)
    Fm, Fs = mean_std(slice_f1s) if slice_f1s else (np.nan,np.nan)
    Am, As = mean_std(slice_accs) if slice_accs else (np.nan,np.nan)

    print("\nTABLE: Cross-validation slice-level results (mean ± std across folds)")
    print("----------------------------------------------------------------")
    print(f"Metric                 Slice-Level")
    print("----------------------------------------------------------------")
    print(f"Accuracy (%)           {Am:.1f} ± {As:.1f}")
    print(f"Macro-F1 (%)           {Fm:.1f} ± {Fs:.1f}")
    print(f"Precision (%)          {Pm:.1f} ± {Ps:.1f}")
    print(f"Recall (%)             {Rm:.1f} ± {Rs:.1f}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
