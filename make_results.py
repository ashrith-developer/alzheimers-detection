#!/usr/bin/env python3
"""
aggregate_cv_metrics.py

Reads per-fold results under --out (folders: fold_0, fold_1, ...)
and prints a paper-style table (mean ± std) with:
  - Accuracy (%)
  - Macro-F1 (%)
  - Precision (%)
  - Recall (%)

It prefers the following in each fold folder (in order):
  1) metrics.json (must contain slice_acc and/or patient_acc; may also contain per-fold macro_f1/precision/recall)
  2) patient_preds.npz or preds.npz  (contains arrays 'y_true' and 'y_pred' for patient-level)
  3) slice_preds.npz (contains 'y_true' and 'y_pred' for slice-level)

If a metric can't be obtained for a level (slice / patient) it prints "N/A" for that cell.
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="./cv_results", help="Folder containing fold_0, fold_1, ...")
parser.add_argument("--fold_prefix", default="fold_", help="Prefix used for fold folders")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

OUT_DIR = args.out
FOLD_PREFIX = args.fold_prefix

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def try_load_npz(path):
    try:
        data = np.load(path, allow_pickle=True)
        return dict(data)
    except Exception:
        return None

def compute_metrics_from_preds(y_true, y_pred):
    # y_true / y_pred are 1D arrays of ints
    if len(y_true) == 0:
        return None
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return {"acc": acc, "macro_f1": macro_f1, "precision": macro_prec, "recall": macro_rec}

def find_fold_dirs(out_dir, prefix="fold_"):
    folds = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith(prefix) and os.path.isdir(os.path.join(out_dir, name)):
            folds.append(os.path.join(out_dir, name))
    return folds

def extract_fold_metrics(fold_dir):
    """
    Returns dictionaries:
      slice_metrics: keys 'acc','macro_f1','precision','recall' values floats in [0,1] or None
      patient_metrics: same keys
    """
    slice_metrics = {k: None for k in ("acc","macro_f1","precision","recall")}
    patient_metrics = {k: None for k in ("acc","macro_f1","precision","recall")}

    # 1) metrics.json
    metrics_path = os.path.join(fold_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            mj = load_json(metrics_path)
            # Accept keys with typical names
            if "slice_acc" in mj: slice_metrics["acc"] = float(mj["slice_acc"])
            if "patient_acc" in mj: patient_metrics["acc"] = float(mj["patient_acc"])
            # optional keys (paper-style names)
            if "patient_macro_f1" in mj: patient_metrics["macro_f1"] = float(mj["patient_macro_f1"])
            if "patient_precision" in mj: patient_metrics["precision"] = float(mj["patient_precision"])
            if "patient_recall" in mj: patient_metrics["recall"] = float(mj["patient_recall"])
            if "slice_macro_f1" in mj: slice_metrics["macro_f1"] = float(mj["slice_macro_f1"])
            if "slice_precision" in mj: slice_metrics["precision"] = float(mj["slice_precision"])
            if "slice_recall" in mj: slice_metrics["recall"] = float(mj["slice_recall"])
        except Exception:
            pass

    # 2) try loading patient preds npz if macro metrics missing
    if any(patient_metrics[k] is None for k in ("macro_f1","precision","recall")):
        for candidate in ("patient_preds.npz", "preds.npz", "patient_predictions.npz"):
            p = os.path.join(fold_dir, candidate)
            if os.path.exists(p):
                d = try_load_npz(p)
                if d and "y_true" in d and "y_pred" in d:
                    m = compute_metrics_from_preds(np.asarray(d["y_true"]), np.asarray(d["y_pred"]))
                    if m:
                        # fill missing patient metrics
                        for k in ("acc","macro_f1","precision","recall"):
                            if patient_metrics[k] is None and k in m:
                                patient_metrics[k] = m[k]
                break

    # 3) try loading slice preds if slice metrics missing
    if any(slice_metrics[k] is None for k in ("macro_f1","precision","recall")):
        for candidate in ("slice_preds.npz", "slice_predictions.npz", "preds_slice.npz"):
            p = os.path.join(fold_dir, candidate)
            if os.path.exists(p):
                d = try_load_npz(p)
                if d and "y_true" in d and "y_pred" in d:
                    m = compute_metrics_from_preds(np.asarray(d["y_true"]), np.asarray(d["y_pred"]))
                    if m:
                        for k in ("acc","macro_f1","precision","recall"):
                            if slice_metrics[k] is None and k in m:
                                slice_metrics[k] = m[k]
                break

    # 4) last resort: maybe metrics.json contains 'patient_report' (sklearn classification_report dict)
    if any(patient_metrics[k] is None for k in ("macro_f1","precision","recall")) and os.path.exists(metrics_path):
        try:
            mj = load_json(metrics_path)
            # some scripts write a full classification_report dict under 'patient_report'
            if "patient_report" in mj and isinstance(mj["patient_report"], dict):
                rep = mj["patient_report"]
                # compute macro averages if present
                if "macro avg" in rep:
                    mac = rep["macro avg"]
                    if patient_metrics["precision"] is None and "precision" in mac:
                        patient_metrics["precision"] = float(mac["precision"])
                    if patient_metrics["recall"] is None and "recall" in mac:
                        patient_metrics["recall"] = float(mac["recall"])
                    if patient_metrics["macro_f1"] is None and "f1-score" in mac:
                        patient_metrics["macro_f1"] = float(mac["f1-score"])
        except Exception:
            pass

    # convert any int -> float and ensure accs are in [0,1]
    for d in (slice_metrics, patient_metrics):
        for k, v in list(d.items()):
            if v is not None:
                try:
                    d[k] = float(v)
                    # If user saved percentages (e.g., 92.4) convert to [0,1] assumption:
                    if d[k] > 1.0:
                        # assume percent -> convert
                        d[k] = d[k] / 100.0
                except Exception:
                    d[k] = None

    return slice_metrics, patient_metrics

def format_pct_meanstd(arr):
    # arr contains values in [0,1] or None. Returns "mean ± std" as percent string or "N/A"
    vals = [v for v in arr if v is not None]
    if not vals:
        return "N/A"
    a = np.array(vals, dtype=float)
    mean = a.mean() * 100.0
    std = a.std(ddof=0) * 100.0
    return f"{mean:.1f} ± {std:.1f}"

def safe_collect(folds):
    slice_accs=[]; pat_accs=[]
    slice_f1=[]; pat_f1=[]
    slice_prec=[]; pat_prec=[]
    slice_rec=[]; pat_rec=[]
    used_folds = 0

    for fd in folds:
        s_m, p_m = extract_fold_metrics(fd)
        if args.verbose:
            print(f"Fold {os.path.basename(fd)} -> slice: {s_m}, patient: {p_m}")
        # we will collect each metric independently
        slice_accs.append(s_m.get("acc"))
        pat_accs.append(p_m.get("acc"))
        slice_f1.append(s_m.get("macro_f1"))
        pat_f1.append(p_m.get("macro_f1"))
        slice_prec.append(s_m.get("precision"))
        pat_prec.append(p_m.get("precision"))
        slice_rec.append(s_m.get("recall"))
        pat_rec.append(p_m.get("recall"))
        used_folds += 1

    return {
        "slice_accs": slice_accs,
        "patient_accs": pat_accs,
        "slice_f1": slice_f1,
        "patient_f1": pat_f1,
        "slice_prec": slice_prec,
        "patient_prec": pat_prec,
        "slice_rec": slice_rec,
        "patient_rec": pat_rec,
        "n_folds": used_folds
    }

def print_table(collected):
    n = collected["n_folds"]
    print("\nTABLE: Cross-validation results (mean ± std across folds)\n")
    # Print header
    header = f"{'Metric':<22} {'Slice-Level':<20} {'Patient-Level':<20}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Accuracy (%)", format_pct_meanstd(collected["slice_accs"]),
                         format_pct_meanstd(collected["patient_accs"])),
        ("Macro-F1 (%)", format_pct_meanstd(collected["slice_f1"]),
                         format_pct_meanstd(collected["patient_f1"])),
        ("Precision (%)", format_pct_meanstd(collected["slice_prec"]),
                          format_pct_meanstd(collected["patient_prec"])),
        ("Recall (%)", format_pct_meanstd(collected["slice_rec"]),
                       format_pct_meanstd(collected["patient_rec"])),
    ]

    for name, s_val, p_val in rows:
        print(f"{name:<22} {s_val:<20} {p_val:<20}")

    print("\nNotes:")
    print(" - Values are mean ± std across folds. Percentages are shown.")
    print(" - If a cell shows 'N/A', the script couldn't find the corresponding per-fold metric.")
    print(" - To compute missing patient-level macro-F1/precision/recall, save per-fold patient predictions as")
    print("     fold_i/patient_preds.npz  with arrays 'y_true' and 'y_pred' (integers).")
    print()

def main():
    if not os.path.isdir(OUT_DIR):
        print("Output directory not found:", OUT_DIR)
        return
    folds = find_fold_dirs(OUT_DIR, prefix=FOLD_PREFIX)
    if not folds:
        print("No fold directories found under", OUT_DIR)
        return

    collected = safe_collect(folds)
    print_table(collected)

if __name__ == "__main__":
    main()
