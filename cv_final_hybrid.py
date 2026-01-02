#!/usr/bin/env python3
"""
cv_final_hybrid.py  (balanced + macroF1 version)

Major changes vs your previous script:
 - class weights are ALWAYS respected (even with label smoothing, via WeightedLabelSmoothingLoss)
 - per-epoch train set is patient-balanced AND class-balanced (oversampling minority-class patients)
 - model selection & early stopping now use patient-level macro F1 instead of plain patient accuracy
"""

import os, argparse, random, json, math, shutil, time, logging
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda import amp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------- ARGS --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./Data", help="Root folder with class subfolders")
parser.add_argument("--out", default="./cv_results_new", help="Output folder")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--img_size", type=int, default=300)
parser.add_argument("--epochs_stage1", type=int, default=18)
parser.add_argument("--epochs_stage2", type=int, default=12)
parser.add_argument("--lr_stage1", type=float, default=2e-4)
parser.add_argument("--lr_stage2", type=float, default=5e-6)
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--ensemble_folds", action="store_true", help="When evaluate, ensemble fold best_stage2 models")
# boosters & model toggles
parser.add_argument("--no_mixup", action="store_true")
parser.add_argument("--no_cutmix", action="store_true")
parser.add_argument("--no_se", action="store_true")
parser.add_argument("--no_mixer", action="store_true")
parser.add_argument("--use_gem", action="store_true", help="Use GeM pooling instead of default avgpool")
parser.add_argument("--use_label_smooth", action="store_true", help="Use label smoothing loss (weighted)")
parser.add_argument("--train_patient_mlp", action="store_true", help="Train patient-level MLP after CV")
parser.add_argument("--max_slices_per_patient", type=int, default=64, help="Max slices sampled per patient per epoch")
parser.add_argument("--tta", action="store_true", help="Enable simple TTA at evaluation (hflip)")
parser.add_argument("--max_val_acc", type=float, default=None)
parser.add_argument("--use_onecycle", action="store_true", help="Use OneCycleLR for training (per-batch stepping)")
args = parser.parse_args()

# -------------------- CONFIG --------------------
DATA_ROOT = args.data
OUT_ROOT = args.out
N_FOLDS = args.folds
BATCH = args.batch
IMG_SIZE = args.img_size
E1 = args.epochs_stage1
E2 = args.epochs_stage2
LR1 = args.lr_stage1
LR2 = args.lr_stage2
PATIENCE = max(2, args.patience)
SEED = args.seed
NUM_WORKERS = args.num_workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

APPLY_MIXUP = not args.no_mixup
APPLY_CUTMIX = not args.no_cutmix
APPLY_SE = not args.no_se
APPLY_MIXER = not args.no_mixer
USE_GEM = args.use_gem
USE_LABEL_SMOOTH = args.use_label_smooth
TRAIN_PATIENT_MLP = args.train_patient_mlp
MAX_SLICES_PER_PATIENT = args.max_slices_per_patient
USE_TTA = args.tta
MAX_VAL_ACC = args.max_val_acc
USE_ONECYCLE = args.use_onecycle

os.makedirs(OUT_ROOT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")
logging.info(f"Batch {BATCH} Img {IMG_SIZE} Max_slices/patient {MAX_SLICES_PER_PATIENT}")
logging.info(f"Mixup:{APPLY_MIXUP} CutMix:{APPLY_CUTMIX} SE:{APPLY_SE} Mixer:{APPLY_MIXER} GeM:{USE_GEM}")
logging.info(f"LabelSmooth:{USE_LABEL_SMOOTH} OneCycle:{USE_ONECYCLE} TTA:{USE_TTA}")
logging.info("Checkpoint selection metric: PATIENT MACRO F1")

# -------------------- TRANSFORMS --------------------
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.88,1.0)),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- HELPERS --------------------
def get_patient_id(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts)>=2 else os.path.splitext(base)[0]

def build_patient_index(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])

    # Binary mapping
    # 0 = Non Demented
    # 1 = Demented (Very mild + Demented)
    binary_map = {
        "Non Demented": 0,
        "Very mild Dementia": 1,
        "Demented": 1
    }

    cls2idx = {k: binary_map[k] for k in classes}

    patient_map = defaultdict(list)

    for c in classes:
        pdir = os.path.join(root, c)
        for f in os.listdir(pdir):
            if not f.lower().endswith((".jpg",".jpeg",".png")):
                continue
            pid = get_patient_id(f)
            patient_map[pid].append(
                (os.path.join(pdir,f), cls2idx[c])
            )

    # Assign patient-level label by majority vote
    patient_label = {}
    for pid, items in patient_map.items():
        lbls = [l for _,l in items]
        patient_label[pid] = Counter(lbls).most_common(1)[0][0]

    # New binary class names
    classes = ["Non Demented", "Demented"]

    return classes, {0:0, 1:1}, patient_map, patient_label


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

# -------------------- MODEL PARTS --------------------
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

class GeM(nn.Module):
    def __init__(self, p=3., eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-2,-1)).pow(1.0/self.p)

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
    def forward(self,x): return self.net(x)

def create_model(num_classes, train_backbone=False):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = train_backbone
    # robustly probe features
    model.eval()
    with torch.no_grad():
        try:
            feat = model.features(torch.zeros(1,3,IMG_SIZE,IMG_SIZE))
        except Exception:
            feat = model.features(torch.zeros(1,3,224,224))
    channels = feat.shape[1]
    if APPLY_SE:
        model.features.add_module("SE_Attention", SEBlock(channels))
    # classifier swap
    try:
        in_f = model.classifier[1].in_features
    except Exception:
        in_f = 1280
    if USE_GEM:
        # customized head: GeM -> flatten -> mixer/linear
        head = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Dropout(0.4),
            MLPMixerHead(in_f, num_classes, drop=0.3) if APPLY_MIXER else nn.Linear(in_f, num_classes)
        )
        model.classifier = head
    else:
        if APPLY_MIXER:
            model.classifier = nn.Sequential(nn.Dropout(0.4), MLPMixerHead(in_f, num_classes, drop=0.3))
        else:
            model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    return model.to(DEVICE)

# -------------------- AUG HELPERS --------------------
def mixup_data(x,y,alpha=1.0):
    if alpha<=0: return x,y,y,1.0
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

def cutmix_data(x,y,alpha=1.0):
    if alpha<=0: return x,y,y,1.0
    lam = np.random.beta(alpha,alpha)
    b,_,H,W = x.size()
    idx = torch.randperm(b).to(x.device)
    cx = np.random.randint(0,W); cy = np.random.randint(0,H)
    cut_w = int(W * np.sqrt(1-lam)); cut_h = int(H * np.sqrt(1-lam))
    x1 = max(cx-cut_w//2,0); y1 = max(cy-cut_h//2,0)
    x2 = min(cx+cut_w//2,W); y2 = min(cy+cut_h//2,H)
    if x2<=x1 or y2<=y1: return mixup_data(x,y,alpha)
    x_clone = x.clone()
    x_clone[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam_adj = 1 - ((x2-x1)*(y2-y1)/(W*H))
    return x_clone, y, y[idx], lam_adj

# -------------------- LOSSES & AMP --------------------
class LabelSmoothingLoss(nn.Module):
    """
    Weighted label smoothing loss.
    If weight is provided, each sample is scaled by weight[target_class].
    """
    def __init__(self, classes, smoothing=0.1, weight=None):
        super().__init__()
        self.cls = classes
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        if weight is not None:
            # register as buffer so device moves are handled automatically if needed
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, x, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        loss_per_sample = torch.sum(-true_dist * self.log_softmax(x), dim=1)
        if self.weight is not None:
            w = self.weight[target]  # shape [N]
            loss_per_sample = loss_per_sample * w
        return loss_per_sample.mean()

scaler = amp.GradScaler()

# -------------------- EVAL / TTA --------------------
def tta_predict(model, imgs, tta_transforms=None):
    model.eval()
    if tta_transforms is None:
        tta_transforms = [lambda x: x, lambda x: torch.flip(x, dims=[3])]
    probs_accum = None
    with torch.no_grad():
        for fn in tta_transforms:
            inp = fn(imgs).to(DEVICE)
            out = model(inp)
            p = nn.Softmax(dim=1)(out).cpu().numpy()
            probs_accum = p if probs_accum is None else probs_accum + p
    return probs_accum / len(tta_transforms)

def evaluate_slices_and_patients(model, loader, classes, use_tta=False):
    model.eval()
    soft = nn.Softmax(dim=1)
    slice_preds, slice_labels, slice_pids, slice_probs = [],[],[],[]
    with torch.no_grad():
        for imgs, labels, pids in tqdm(loader, desc="Eval(slices)", leave=False):
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            if use_tta:
                probs = tta_predict(model, imgs)
            else:
                with amp.autocast():
                    out = model(imgs)
                    probs = soft(out).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            slice_preds += preds; slice_labels += labels.cpu().numpy().tolist(); slice_pids += pids; slice_probs += probs.tolist()
    # patient aggregation by average prob
    patient_probs = defaultdict(list); patient_true = {}
    for pid, lab, prob in zip(slice_pids, slice_labels, slice_probs):
        patient_probs[pid].append(prob)
        if pid not in patient_true: patient_true[pid] = lab
    patient_preds=[]; patient_labels=[]
    for pid, probs in patient_probs.items():
        avg = np.mean(np.stack(probs, axis=0), axis=0)
        patient_preds.append(int(np.argmax(avg))); patient_labels.append(int(patient_true[pid]))
    slice_acc = sum(int(a==b) for a,b in zip(slice_preds, slice_labels))/max(1,len(slice_labels))
    patient_acc = sum(int(a==b) for a,b in zip(patient_preds, patient_labels))/max(1,len(patient_labels))
    return {"slice":{"acc":slice_acc}, "patient":{"acc":patient_acc},
            "slice_preds":slice_preds, "slice_labels":slice_labels,
            "patient_preds":patient_preds, "patient_labels":patient_labels}

# -------------------- TRAIN LOOP (OneCycle optional) --------------------
def train_one_epoch(model, loader, optimizer, epoch_idx, mix_prob=0.45, alpha=0.8, scheduler=None, step_scheduler_per_batch=False):
    model.train(); running_loss=0.0; total=0; correct=0
    pbar = tqdm(loader, desc=f"Train E{epoch_idx+1}", ncols=120)
    for imgs, labels, _ in pbar:
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
        optimizer.zero_grad()
        if APPLY_MIXUP and APPLY_CUTMIX and random.random() < mix_prob:
            if random.random() < 0.5 and APPLY_MIXUP:
                imgs_m, y1, y2, lam = mixup_data(imgs, labels, alpha)
            elif APPLY_CUTMIX:
                imgs_m, y1, y2, lam = cutmix_data(imgs, labels, alpha)
            else:
                imgs_m, y1, y2, lam = imgs, labels, labels, 1.0
            with amp.autocast():
                out = model(imgs_m)
                loss = lam * crit(out, y1) + (1-lam) * crit(out, y2)
        else:
            with amp.autocast():
                out = model(imgs)
                loss = crit(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        if scheduler is not None and step_scheduler_per_batch:
            try:
                scheduler.step()
            except Exception:
                pass
        running_loss += float(loss.item()) * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})
    return running_loss/max(1,total), correct/max(1,total)

# -------------------- CHECKPOINT --------------------
def save_ckpt(path, model, optimizer, scheduler, epoch, stage, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"epoch": epoch, "stage": stage, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
             "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
             "scaler_state_dict": scaler.state_dict(), "best_val_acc": best_val}
    torch.save(state, path)

def load_ckpt(path, model, optimizer=None, scheduler=None):
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    if optimizer is not None and ck.get("optimizer_state_dict"):
        try: optimizer.load_state_dict(ck["optimizer_state_dict"])
        except Exception as e: logging.warning("optimizer restore failed: %s", e)
    if scheduler is not None and ck.get("scheduler_state_dict"):
        try: scheduler.load_state_dict(ck["scheduler_state_dict"])
        except Exception as e: logging.warning("scheduler restore failed: %s", e)
    try: scaler.load_state_dict(ck.get("scaler_state_dict", {}))
    except: pass
    return ck

# -------------------- PATIENT / CLASS BALANCED SAMPLERS --------------------
def build_epoch_train_items(patient_map, train_pids, max_per_patient=64, ordered_sampling=True):
    items=[]
    for pid in train_pids:
        entries = patient_map.get(pid, [])
        if not entries: continue
        if len(entries) > max_per_patient:
            if ordered_sampling:
                step = len(entries) / max_per_patient
                sampled = [entries[int(i*step)] for i in range(max_per_patient)]
            else:
                sampled = random.sample(entries, max_per_patient)
        else:
            sampled = entries[:]
        for path,lbl in sampled:
            items.append((path, lbl, pid))
    random.shuffle(items)
    return items

def make_class_balanced_pids(train_pids, patient_label):
    """
    Oversample patients so that each class contributes roughly the same
    number of patients per epoch. This is critical for minority classes.
    """
    per_class = defaultdict(list)
    for pid in train_pids:
        c = patient_label[pid]
        per_class[c].append(pid)
    max_len = max(len(v) for v in per_class.values())
    balanced = []
    for c, pids in per_class.items():
        if len(pids) == 0:
            continue
        repeat = math.ceil(max_len / len(pids))
        tmp = (pids * repeat)[:max_len]
        balanced.extend(tmp)
    random.shuffle(balanced)
    return balanced


# -------------------- GRADCAM helper --------------------
def find_target_layer(model):
    for layer in reversed(list(model.features.children())):
        if isinstance(layer, nn.Conv2d): return layer
        if isinstance(layer, nn.Sequential):
            for sub in reversed(list(layer.children())):
                if isinstance(sub, nn.Conv2d): return sub
    return model.features[-1]

# -------------------- CROSS-VAL DRIVER --------------------
def run_cv():
    classes, cls2idx, patient_map, patient_label = build_patient_index(DATA_ROOT)
    num_classes = len(classes)
    logging.info("Classes: %s", classes)
    patient_ids = sorted(list(patient_map.keys()))
    labels = np.array([patient_label[pid] for pid in patient_ids])
    pid2idx = {pid:i for i,pid in enumerate(patient_ids)}
    group_idxs = np.array([pid2idx[pid] for pid in patient_ids])

    # patient-count based class weights
    patient_counts = Counter([patient_label[pid] for pid in patient_ids])
    cls_counts = np.array([patient_counts.get(i,0) for i in range(num_classes)], dtype=float)
    cls_counts = np.where(cls_counts==0, 1.0, cls_counts)
    cls_weights = (cls_counts.max() / cls_counts)
    # optional: normalize so mean weight ~1
    cls_weights = cls_weights / cls_weights.mean()
    cls_weights_t = torch.tensor(cls_weights, dtype=torch.float).to(DEVICE)
    logging.info("Patient counts per class: %s", dict(patient_counts))
    logging.info("Class weights (patient-based, normalized): %s", cls_weights.tolist())

    # choose criterion (GLOBAL)
    global crit
    if USE_LABEL_SMOOTH:
        crit = LabelSmoothingLoss(num_classes, smoothing=0.1, weight=cls_weights_t)
        logging.info("Using weighted label smoothing loss")
    else:
        crit = nn.CrossEntropyLoss(weight=cls_weights_t)
        logging.info("Using weighted CrossEntropy loss")

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold = 0; summary = {}
    for tv_idx, test_idx in gkf.split(patient_ids, labels, groups=group_idxs):
        train_val_pids = [patient_ids[i] for i in tv_idx]
        test_pids = [patient_ids[i] for i in test_idx]
        # split val from train_val (patient-wise)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
        idxs = list(range(len(train_val_pids)))
        tr_idx, val_idx = next(gss.split(idxs, groups=idxs))
        train_pids = [train_val_pids[i] for i in tr_idx]; val_pids = [train_val_pids[i] for i in val_idx]
        logging.info("\n--- Fold %d: train %d val %d test %d ---", fold, len(train_pids), len(val_pids), len(test_pids))

        def gather(pids):
            lst=[]
            for pid in pids:
                for path,lbl in patient_map[pid]:
                    lst.append((path,lbl,pid))
            return lst

        val_items = gather(val_pids); test_items = gather(test_pids)
        val_ds = SliceDataset(val_items, transform=val_tf)
        test_ds = SliceDataset(test_items, transform=val_tf)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # model and optimizer for stage1
        model = create_model(num_classes, train_backbone=False)
        for p in model.classifier.parameters(): p.requires_grad = True
        base_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.AdamW(base_params, lr=LR1, weight_decay=1e-5)
        # scheduler: either OneCycle or ReduceLROnPlateau (we will create per-stage)
        scheduler = None

        ckpt_dir = os.path.join(OUT_ROOT, f"fold_{fold}")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
        best_stage1_path = os.path.join(ckpt_dir, "best_stage1.pth")
        best_stage2_path = os.path.join(ckpt_dir, "best_stage2.pth")
        best_metric = 0.0  # this is patient-level macro F1
        start_stage = 1; start_epoch = 0

        if args.resume and os.path.exists(ckpt_path):
            try:
                ck = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ck["model_state_dict"])
                if ck.get("optimizer_state_dict"):
                    try: optimizer.load_state_dict(ck["optimizer_state_dict"])
                    except: pass
                if ck.get("scheduler_state_dict") and scheduler is not None:
                    try: scheduler.load_state_dict(ck["scheduler_state_dict"])
                    except: pass
                try: scaler.load_state_dict(ck.get("scaler_state_dict", {}))
                except: pass
                start_stage = ck.get("stage",1); start_epoch = ck.get("epoch",0)+1
                best_metric = ck.get("best_val_acc", 0.0)
                logging.info("Resuming fold %d stage %d from epoch %d (best_metric=%.4f)", fold, start_stage, start_epoch, best_metric)
            except Exception as e:
                logging.warning("Resume failed: %s", e)

        # --------------- STAGE 1 ---------------
        if start_stage <= 1:
            early = 0
            for e in range(start_epoch, E1):
                # class-balanced patient sampling
                balanced_train_pids = make_class_balanced_pids(train_pids, patient_label)
                # build epoch train items (patient-balanced)
                train_items_epoch = build_epoch_train_items(
                    patient_map, balanced_train_pids,
                    max_per_patient=MAX_SLICES_PER_PATIENT, ordered_sampling=True
                )
                train_ds = SliceDataset(train_items_epoch, transform=train_tf)
                train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

                # optionally create OneCycleLR per epoch (if requested) for stage1
                if USE_ONECYCLE:
                    steps_per_epoch = max(1, math.ceil(len(train_items_epoch) / BATCH))
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=LR1,
                        epochs=E1, steps_per_epoch=steps_per_epoch,
                        pct_start=0.2, anneal_strategy='cos',
                        div_factor=25.0, final_div_factor=1e4
                    )
                    step_scheduler_per_batch = True
                else:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=0.5, patience=max(2,PATIENCE//2)
                    )
                    step_scheduler_per_batch = False

                # train epoch
                train_one_epoch(model, train_loader, optimizer, e, scheduler=scheduler, step_scheduler_per_batch=step_scheduler_per_batch)

                # patient-level validation
                eval_val = evaluate_slices_and_patients(model, val_loader, classes, use_tta=USE_TTA)
                vacc = eval_val['patient']['acc']
                f1_macro = f1_score(eval_val['patient_labels'], eval_val['patient_preds'], average='macro')
                logging.info("[S1] Epoch %d patient_acc %.4f patient_macroF1 %.4f", e, vacc, f1_macro)

                # scheduler step
                if USE_ONECYCLE:
                    # OneCycleLR stepped per batch already
                    pass
                else:
                    scheduler.step(1.0 - f1_macro)  # use macroF1 as metric for LR scheduling

                # early stopping / checkpoint metric: macro F1
                metric = f1_macro

                # cap check & save best
                if MAX_VAL_ACC is not None and vacc >= MAX_VAL_ACC:
                    logging.warning("Reached max_val_acc %.4f >= %.4f; skipping saving", vacc, MAX_VAL_ACC)
                    early += 1
                else:
                    if metric > best_metric:
                        os.makedirs(os.path.dirname(best_stage1_path), exist_ok=True)
                        torch.save({"model_state_dict": model.state_dict()}, best_stage1_path)
                        best_metric = metric; early = 0
                        logging.info("Saved best_stage1 (metric=%.4f)", best_metric)
                    else:
                        early += 1

                save_ckpt(ckpt_path, model, optimizer, scheduler, e, stage=1, best_val=best_metric)
                if early >= PATIENCE:
                    logging.info("Early stop stage1 (no macroF1 improvement)")
                    break

        # --------------- STAGE 2 ---------------
        logging.info("Stage 2: unfreeze top blocks & fine-tune")
        if os.path.exists(best_stage1_path):
            st = torch.load(best_stage1_path, map_location=DEVICE)
            try: model.load_state_dict(st["model_state_dict"])
            except: model.load_state_dict(st)

        children = list(model.features.children())
        n_unfreeze = min(6, len(children))
        for layer in children[-n_unfreeze:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in model.classifier.parameters(): p.requires_grad = True

        opt2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR2, weight_decay=1e-6)
        sched2 = None

        # === SAFER resume logic for stage2 ===
        start_epoch2 = 0
        best_metric2 = best_metric  # carry over
        if args.resume and os.path.exists(ckpt_path):
            try:
                ck = torch.load(ckpt_path, map_location=DEVICE)
                if ck.get("stage",1) == 2:
                    try:
                        model.load_state_dict(ck["model_state_dict"])
                    except Exception as e:
                        logging.warning("model restore during stage2 resume failed: %s", e)
                    if ck.get("optimizer_state_dict"):
                        try:
                            opt2.load_state_dict(ck["optimizer_state_dict"])
                        except Exception as e:
                            logging.warning("opt2 restore failed during stage2 resume: %s", e)
                    if ck.get("scheduler_state_dict") and sched2 is not None:
                        try:
                            sched2.load_state_dict(ck["scheduler_state_dict"])
                        except Exception as e:
                            logging.warning("sched2 restore failed during stage2 resume: %s", e)
                    try:
                        scaler.load_state_dict(ck.get("scaler_state_dict", {}))
                    except Exception as e:
                        logging.warning("scaler restore failed during stage2 resume: %s", e)
                    start_epoch2 = ck.get("epoch",0)+1
                    best_metric2 = ck.get("best_val_acc", best_metric2)
                    logging.info("Resuming stage2 from %d (best_metric2=%.4f)", start_epoch2, best_metric2)
            except Exception as e:
                logging.warning("resume stage2 failed: %s", e)

        early2 = 0
        for e in range(start_epoch2, E2):
            balanced_train_pids = make_class_balanced_pids(train_pids, patient_label)
            train_items_epoch = build_epoch_train_items(
                patient_map, balanced_train_pids,
                max_per_patient=MAX_SLICES_PER_PATIENT, ordered_sampling=True
            )
            train_ds = SliceDataset(train_items_epoch, transform=train_tf)
            train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

            if USE_ONECYCLE:
                steps_per_epoch = max(1, math.ceil(len(train_items_epoch) / BATCH))
                sched2 = torch.optim.lr_scheduler.OneCycleLR(
                    opt2, max_lr=LR2*10 if LR2>0 else LR1,
                    epochs=E2, steps_per_epoch=steps_per_epoch,
                    pct_start=0.2, anneal_strategy='cos',
                    div_factor=25.0, final_div_factor=1e4
                )
                step_scheduler_per_batch = True
            else:
                sched2 = optim.lr_scheduler.ReduceLROnPlateau(
                    opt2, factor=0.5, patience=max(2,PATIENCE//2)
                )
                step_scheduler_per_batch = False

            train_one_epoch(model, train_loader, opt2, e, scheduler=sched2, step_scheduler_per_batch=step_scheduler_per_batch)
            eval_val = evaluate_slices_and_patients(model, val_loader, classes, use_tta=USE_TTA)
            vacc = eval_val['patient']['acc']
            f1_macro = f1_score(eval_val['patient_labels'], eval_val['patient_preds'], average='macro')
            logging.info("[S2] Epoch %d patient_acc %.4f patient_macroF1 %.4f", e, vacc, f1_macro)

            if USE_ONECYCLE:
                pass
            else:
                try:
                    sched2.step(1.0 - f1_macro)
                except Exception:
                    pass

            metric2 = f1_macro

            if MAX_VAL_ACC is not None and vacc >= MAX_VAL_ACC:
                logging.warning("Reached max_val_acc %.4f >= %.4f; skipping save", vacc, MAX_VAL_ACC)
                early2 += 1
            else:
                if metric2 > best_metric2:
                    os.makedirs(os.path.dirname(best_stage2_path), exist_ok=True)
                    torch.save({"model_state_dict": model.state_dict()}, best_stage2_path)
                    best_metric2 = metric2; early2 = 0
                    logging.info("Saved best_stage2 (metric=%.4f)", best_metric2)
                else:
                    early2 += 1

            save_ckpt(ckpt_path, model, opt2, sched2, e, stage=2, best_val=best_metric2)
            if early2 >= PATIENCE:
                logging.info("Early stop stage2 (no macroF1 improvement)")
                break

        # --------------- EVAL ---------------
        best_model_path = best_stage2_path if os.path.exists(best_stage2_path) else ckpt_path
        logging.info("Loading best for eval: %s", best_model_path)
        model_eval = create_model(num_classes, train_backbone=False)
        ck = torch.load(best_model_path, map_location=DEVICE)
        if isinstance(ck, dict) and "model_state_dict" in ck:
            model_eval.load_state_dict(ck["model_state_dict"])
        else:
            model_eval.load_state_dict(ck)
        model_eval.to(DEVICE)
        eval_res = evaluate_slices_and_patients(model_eval, test_loader, classes, use_tta=USE_TTA)
        logging.info("Fold %d patient-acc %.4f slice-acc %.4f", fold, eval_res['patient']['acc'], eval_res['slice']['acc'])

        # grad-cam save for a few images
        try:
            target_layer = find_target_layer(model_eval)
            cam = GradCAM(model=model_eval, target_layers=[target_layer])
            for imgs, labs, pids in test_loader:
                imgs_cpu = imgs.clone()
                imgs = imgs.to(DEVICE)
                grayscale = cam(input_tensor=imgs)
                k = min(3, imgs.size(0))
                fold_dir = os.path.join(OUT_ROOT, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                for i in range(k):
                    g = grayscale[i]
                    img_np = imgs_cpu[i].permute(1,2,0).numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    vis = show_cam_on_image(img_np, g, use_rgb=True)
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(fold_dir, f"gradcam_{i}.png"), vis)
                break
        except Exception as e:
            logging.warning("Grad-CAM failed: %s", e)

        # save metrics
        os.makedirs(os.path.join(OUT_ROOT, f"fold_{fold}"), exist_ok=True)
        with open(os.path.join(OUT_ROOT, f"fold_{fold}", "metrics.json"), "w") as f:
            json.dump({"patient_acc": eval_res['patient']['acc'], "slice_acc": eval_res['slice']['acc']}, f, indent=2)
        summary[f"fold_{fold}"] = {"patient_acc": eval_res['patient']['acc'], "slice_acc": eval_res['slice']['acc']}
        fold += 1

    # overall summary
    with open(os.path.join(OUT_ROOT, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("CV done. results: %s", OUT_ROOT)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError("Data root not found: " + DATA_ROOT)
    if args.evaluate:
        evaluate_mode()
    else:
        run_cv()
