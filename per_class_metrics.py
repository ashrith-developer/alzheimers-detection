#!/usr/bin/env python3
"""
per_class_metrics.py

Evaluation script matched to cv_final_hybrid.py (SE + MLP-Mixer, binary mapping).
Loads best_stage2.pth / checkpoint.pth / best_stage1.pth per fold and computes
patient-level classification report averaged across GroupKFold splits.

Usage:
 python per_class_metrics.py --data ./Data --out ./cv_results_new --binary --tta --num_workers 0
"""

import os
import argparse
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda import amp

# ----------------- ARGS -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./Data", help="Root folder with class subfolders")
parser.add_argument("--out", default="./cv_results_new", help="Folder with fold_x subfolders (from training)")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--img_size", type=int, default=300)
parser.add_argument("--tta", action="store_true", help="Use simple TTA (hflip)")
parser.add_argument("--binary", action="store_true", help="Use binary mapping (Non Demented vs Demented)")
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (use 0 on Windows)")
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

# ----------------- HELPERS -----------------
def get_patient_id(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else os.path.splitext(base)[0]

def build_patient_index(root, binary=False):
    folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if binary:
        # mapping used in your training script
        mapping = {
            "Non Demented": 0,
            "Very mild Dementia": 1,
            "Demented": 1
        }
        # warn if folders differ but still attempt mapping
        missing = set(mapping.keys()) - set(folders)
        if missing:
            print("Warning: data folder missing expected names for binary mapping:", missing)
            print("Found folders:", folders)
        cls2idx = {f: mapping.get(f, 1) for f in folders}  # fallback to Demented(1)
        classes = ["Non Demented", "Demented"]
    else:
        cls2idx = {c: i for i, c in enumerate(folders)}
        classes = folders

    patient_map = defaultdict(list)
    for c in folders:
        pdir = os.path.join(root, c)
        for f in os.listdir(pdir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            pid = get_patient_id(f)
            patient_map[pid].append((os.path.join(pdir, f), cls2idx[c]))

    patient_label = {}
    for pid, items in patient_map.items():
        lbls = [l for _, l in items]
        patient_label[pid] = Counter(lbls).most_common(1)[0][0]

    return classes, cls2idx, patient_map, patient_label

class SliceDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label, pid = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, pid

# ----------------- MODEL PARTS (mirror training) -----------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
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

def create_model(num_classes: int):
    # EfficientNet V2 S with ImageNet weights (same as training)
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.eval()
    # probe features for channels
    with torch.no_grad():
        try:
            feat = model.features(torch.zeros(1,3,args.img_size,args.img_size))
        except Exception:
            feat = model.features(torch.zeros(1,3,224,224))
    channels = feat.shape[1]
    # add SE block (training appended SE_Attention)
    model.features.add_module("SE_Attention", SEBlock(channels))
    # find classifier in_features
    try:
        in_f = model.classifier[1].in_features
    except Exception:
        in_f = 1280
    # use MLP-Mixer head (training default)
    model.classifier = nn.Sequential(nn.Dropout(0.4), MLPMixerHead(in_f, num_classes, drop=0.3))
    return model.to(DEVICE)

# ----------------- CHECKPOINT LOADER (robust) -----------------
def inspect_state_dict_for_out_dim(state):
    # find plausible final linear weight with small out_dim (<= 50)
    candidates = []
    for k, v in state.items():
        # we only consider 2D weight-like tensors
        try:
            shape = getattr(v, "shape", None)
            if shape is None: continue
            if len(shape) != 2: continue
            out_dim = int(shape[0])
            # heuristics: final linear usually has small output_dim (<=50)
            if out_dim <= 50:
                candidates.append((k, out_dim))
        except Exception:
            continue
    if not candidates:
        return None, None
    # prefer keys containing 'classifier', 'net', 'head'
    candidates = sorted(candidates, key=lambda x: (
        ('classifier' in x[0]) or ('head' in x[0]) or ('net' in x[0]),
        x[0].count('.')
    ), reverse=True)
    return candidates[0]  # (key, out_dim)

def load_best_model_for_fold(fold_idx, num_classes):
    fold_dir = os.path.join(args.out, f"fold_{fold_idx}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Missing folder: {fold_dir}")
    candidates = ["best_stage2.pth", "checkpoint.pth", "best_stage1.pth"]
    chosen = None
    for fn in candidates:
        p = os.path.join(fold_dir, fn)
        if os.path.exists(p):
            chosen = p
            break
    if chosen is None:
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    print(f"Fold {fold_idx}: loading {chosen}")
    ck = torch.load(chosen, map_location=DEVICE)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck

    # quick detection of classifier out-dim
    cand_key, out_dim = inspect_state_dict_for_out_dim(state)
    if cand_key is not None:
        print(f"Detected candidate classifier weight: '{cand_key}' out_dim={out_dim}")
        if out_dim != num_classes:
            raise RuntimeError(
                f"Checkpoint has classifier output dim = {out_dim} but requested num_classes = {num_classes}.\n"
                "This usually means the checkpoint was trained with a different class mapping. "
                "Point --out to the matching results folder or switch binary flag accordingly."
            )
    else:
        print("Warning: could not auto-detect small final-linear weight in checkpoint; proceeding with cautious load.")

    model = create_model(num_classes)

    # try strict load first
    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint via strict load.")
        return model
    except RuntimeError as e:
        # strict failed: fallback to partial load (intersection)
        print("Strict load failed:", str(e))
        print("Attempting to load matching keys (partial load). This may still be OK if mismatch is only naming/extra keys.")
        model_state = model.state_dict()
        filtered = {}
        missing_keys = []
        unexpected_keys = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
        # load filtered
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        # print diagnostics
        print(f"Loaded {len(filtered)} matching parameters out of {len(state)} checkpoint params.")
        # show a few mismatches if helpful
        ck_keys = set(state.keys())
        model_keys = set(model.state_dict().keys())
        extra = ck_keys - model_keys
        if extra:
            print("Checkpoint has extra keys (first 10 shown):", list(extra)[:10])
        missing = model_keys - ck_keys
        if missing:
            print("Model has keys missing in checkpoint (first 10 shown):", list(missing)[:10])
        return model

# ----------------- EVAL / TTA -----------------
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

def evaluate_slices_and_patients(model, loader, use_tta=False):
    model.eval()
    soft = nn.Softmax(dim=1)
    slice_preds, slice_labels, slice_pids, slice_probs = [], [], [], []
    with torch.no_grad():
        for imgs, labels, pids in tqdm(loader, desc="Eval(slices)", leave=False):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            if use_tta:
                probs = tta_predict(model, imgs)
            else:
                with amp.autocast():
                    out = model(imgs)
                    probs = soft(out).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            slice_preds += preds; slice_labels += labels.cpu().numpy().tolist(); slice_pids += pids; slice_probs += probs.tolist()
    # patient aggregation
    patient_probs = defaultdict(list); patient_true = {}
    for pid, lab, prob in zip(slice_pids, slice_labels, slice_probs):
        patient_probs[pid].append(prob)
        if pid not in patient_true: patient_true[pid] = lab
    patient_preds=[]; patient_labels=[]
    for pid, probs in patient_probs.items():
        avg = np.mean(np.stack(probs, axis=0), axis=0)
        patient_preds.append(int(np.argmax(avg))); patient_labels.append(int(patient_true[pid]))
    return {"slice":{"preds":slice_preds,"labels":slice_labels}, "patient":{"preds":patient_preds,"labels":patient_labels}}

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
    fold_reports = []

    for fold_idx, (tv_idx, test_idx) in enumerate(gkf.split(patient_ids, labels, groups=group_idxs)):
        print(f"\n======================\nFold {fold_idx}\n======================")
        test_pids = [patient_ids[i] for i in test_idx]
        test_items=[]
        for pid in test_pids:
            for p,l in patient_map[pid]:
                test_items.append((p,l,pid))
        ds = SliceDataset(test_items, transform=val_tf)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(DEVICE=="cuda"))

        model = load_best_model_for_fold(fold_idx, num_classes)
        model.to(DEVICE); model.eval()

        res = evaluate_slices_and_patients(model, loader, use_tta=args.tta)
        y_true = res["patient"]["labels"]
        y_pred = res["patient"]["preds"]

        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        conf = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        fold_reports.append(report)

        print("Patient-level classification report (this fold):")
        for cls_name in classes:
            r = report[cls_name]
            print(f"  {cls_name:20s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")
        mac = report["macro avg"]
        print(f"  {'Macro Avg.':20s}  P={mac['precision']:.3f}  R={mac['recall']:.3f}  F1={mac['f1-score']:.3f}")

    # average across folds
    print("\n===============================================")
    print("PER-CLASS METRICS (PATIENT LEVEL, AVERAGED ACROSS FOLDS)")
    print("===============================================")
    acc = {cls: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for cls in classes}
    acc["Macro Avg."] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for rep in fold_reports:
        for cls_name in classes:
            r = rep[cls_name]
            acc[cls_name]["precision"] += r["precision"]
            acc[cls_name]["recall"] += r["recall"]
            acc[cls_name]["f1"] += r["f1-score"]
        mac = rep["macro avg"]
        acc["Macro Avg."]["precision"] += mac["precision"]
        acc["Macro Avg."]["recall"] += mac["recall"]
        acc["Macro Avg."]["f1"] += mac["f1-score"]
    nfolds = len(fold_reports)
    print(f"\nTable (like in the paper), averaged over {nfolds} folds:\n")
    print(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("-" * 60)
    for cls_name in classes:
        p = acc[cls_name]["precision"] / nfolds
        r = acc[cls_name]["recall"] / nfolds
        f1 = acc[cls_name]["f1"] / nfolds
        print(f"{cls_name:20s} {p:10.3f} {r:10.3f} {f1:10.3f}")
    p = acc["Macro Avg."]["precision"] / nfolds
    r = acc["Macro Avg."]["recall"] / nfolds
    f1 = acc["Macro Avg."]["f1"] / nfolds
    print("-" * 60)
    print(f"{'Macro Avg.':20s} {p:10.3f} {r:10.3f} {f1:10.3f}")

if __name__ == "__main__":
    main()
