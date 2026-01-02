#!/usr/bin/env python3
"""
cv_nsmnet_full_final.py — NSM-Net final, ready-to-run.

This is a self-contained single-file script intended as a drop-in replacement
for the previous main.py. Changes and robustness additions:
 - BalancedClassPatientSampler for per-batch class coverage when possible
 - FocalLoss to handle heavy class imbalance
 - Robust checkpoint loader that partially loads matching keys/shapes
 - Improved GradCAM init (supports different library versions)
 - CSV logging, per-fold confusion matrix and train/val curve plots
 - Optional precompute DCT caching
 - Prints helpful debug/info statements

Usage example:
  python cv_nsmnet_full_final.py --data ./Data --out ./cv_out --batch_patients 4 --slices_per_patient 8

Make sure you have the required packages installed in your env:
  torch, torchvision, timm, numpy, pillow, matplotlib, pandas, scipy, sklearn, tqdm

"""

import os, sys, argparse, random, json, math, time, csv
from collections import defaultdict, Counter
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from scipy.fftpack import dct
import timm
from sklearn.metrics import confusion_matrix, classification_report
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except Exception:
    HAS_GRADCAM = False

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./Data", help="Root folder with class subfolders (sliced images)")
parser.add_argument("--out", default="./cv_out", help="Output folder for checkpoints/logs")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--batch_patients", type=int, default=4)
parser.add_argument("--slices_per_patient", type=int, default=8)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--epochs_stage1", type=int, default=8)
parser.add_argument("--epochs_stage2", type=int, default=6)
parser.add_argument("--lr_stage1", type=float, default=2e-4)
parser.add_argument("--lr_stage2", type=float, default=1e-5)
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--no_mcs", action="store_true")
parser.add_argument("--no_hfg", action="store_true")
parser.add_argument("--no_asiw", action="store_true")
parser.add_argument("--precompute_dct", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--max_val_acc", type=float, default=None)
args = parser.parse_args()

# -------------------------
# Config & environment
# -------------------------
DATA_ROOT = args.data
OUT_ROOT = args.out
N_FOLDS = args.folds
P = args.batch_patients
S = args.slices_per_patient
IMG_SIZE = args.img_size
E1 = args.epochs_stage1
E2 = args.epochs_stage2
LR1 = args.lr_stage1
LR2 = args.lr_stage2
PATIENCE = args.patience
SEED = args.seed
NUM_WORKERS = args.num_workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_MCSM = not args.no_mcs
USE_HFG = not args.no_hfg
USE_ASIW = not args.no_asiw
PRECOMP_DCT = args.precompute_dct
MAX_VAL_ACC = args.max_val_acc

os.makedirs(OUT_ROOT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

print("DEVICE:", DEVICE, "P:", P, "S:", S, "MCSM:", USE_MCSM, "HFG:", USE_HFG, "ASIW:", USE_ASIW)

# -------------------------
# Transforms
# -------------------------
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0)),
    T.ColorJitter(brightness=0.08, contrast=0.08),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -------------------------
# Helpers
# -------------------------

def get_patient_id_from_filename(fname):
    base = os.path.basename(fname)
    toks = base.split('_')
    if len(toks) >= 2:
        return toks[0] + "_" + toks[1]
    return os.path.splitext(base)[0]


def build_patient_index(data_root):
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))])
    cls2idx = {c:i for i,c in enumerate(classes)}
    patient_map = defaultdict(list)
    for c in classes:
        pdir = os.path.join(data_root,c)
        for f in sorted(os.listdir(pdir)):
            path = os.path.join(pdir,f)
            if not os.path.isfile(path): continue
            if not f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif')): continue
            pid = get_patient_id_from_filename(f)
            patient_map[pid].append((path, cls2idx[c]))
    patient_label = {pid: Counter([lbl for _,lbl in items]).most_common(1)[0][0] for pid,items in patient_map.items()}
    return classes, cls2idx, patient_map, patient_label

# -------------------------
# DCT helpers
# -------------------------

def compute_2d_dct_np(img_arr, k=256):
    c = dct(dct(img_arr.T, norm='ortho').T, norm='ortho')
    sq = int(math.sqrt(k))
    block = c[:sq, :sq].flatten()
    if block.shape[0] < k:
        block = np.pad(block, (0, k - block.shape[0]))
    return block[:k].astype(np.float32)


def precompute_dct_for_dataset(patient_map, k=256, img_size=IMG_SIZE, force=False):
    print("Precomputing DCT for dataset (k=%d)..." % k)
    count=0
    for pid, items in tqdm(patient_map.items()):
        for path,_ in items:
            cache = path + ".dct.npy"
            if os.path.exists(cache) and not force: continue
            img = Image.open(path).convert('L').resize((img_size,img_size))
            arr = np.array(img).astype(np.float32)/255.0
            vec = compute_2d_dct_np(arr, k=k)
            np.save(cache, vec)
            count += 1
    print("Saved DCT for %d images" % count)

# -------------------------
# Balanced patient sampler
# -------------------------
class BalancedClassPatientSampler:
    def __init__(self, patient_map, patient_label, patients_per_batch=4, slices_per_patient=8):
        self.p2files = {pid:[p for p,_ in items] for pid,items in patient_map.items()}
        self.p2label = patient_label
        self.class2pids = defaultdict(list)
        for pid,lbl in patient_label.items():
            if pid in self.p2files:
                self.class2pids[lbl].append(pid)
        self.P = patients_per_batch
        self.S = slices_per_patient
        self.pids = list(self.p2files.keys())
        cls_counts = {cls: max(1,len(pids)) for cls,pids in self.class2pids.items()}
        self.fallback_weights = {cls: 1.0/float(cls_counts[cls]) for cls in cls_counts}

    def __len__(self):
        return math.ceil(len(self.pids) / self.P)

    def __iter__(self):
        while True:
            batch_pids = []
            classes = list(self.class2pids.keys())
            random.shuffle(classes)
            for cls in classes:
                if len(batch_pids) >= self.P: break
                if self.class2pids[cls]:
                    batch_pids.append(random.choice(self.class2pids[cls]))
            while len(batch_pids) < self.P:
                cls_choices = list(self.class2pids.keys())
                weights = [self.fallback_weights[c] for c in cls_choices]
                chosen_cls = random.choices(cls_choices, weights=weights, k=1)[0]
                batch_pids.append(random.choice(self.class2pids[chosen_cls]))

            batch_files = []
            for pid in batch_pids:
                files = self.p2files[pid]
                if len(files) >= self.S:
                    sel = random.sample(files, self.S)
                else:
                    sel = files + random.choices(files, k=(self.S - len(files)))
                batch_files.extend(sel)
            batch_labels = [self.p2label[pid] for pid in batch_pids]
            yield batch_files, batch_pids, batch_labels

# -------------------------
# Image loader
# -------------------------

def load_image_and_optional_dct(path, tfm, precompute_dct):
    img = Image.open(path).convert('RGB')
    img_t = tfm(img)
    dct_vec = None
    if precompute_dct:
        cache = path + ".dct.npy"
        if os.path.exists(cache):
            dct_vec = torch.from_numpy(np.load(cache)).float()
    return img_t, dct_vec

# -------------------------
# Model building blocks
# -------------------------
class SimpleGCNLayer(nn.Module):
    def __init__(self,in_f,out_f): super().__init__(); self.lin=nn.Linear(in_f,out_f)
    def forward(self,x,adj):
        h = torch.matmul(adj, x)
        return F.relu(self.lin(h))

class GCNEncoder(nn.Module):
    def __init__(self, node_feat_dim=36, hidden=64, out_dim=128, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_d = node_feat_dim
        for i in range(layers):
            outd = hidden if i < layers-1 else out_dim
            self.layers.append(SimpleGCNLayer(in_d, outd))
            in_d = outd
    def forward(self, node_feats, adj):
        h = node_feats
        for l in self.layers:
            h = l(h, adj)
        return h.mean(dim=1)

class SpectralNet(nn.Module):
    def __init__(self, in_k=256, out_d=128):
        super().__init__()
        self.fc1 = nn.Linear(in_k, 256)
        self.fc2 = nn.Linear(256, out_d)
        self.act = nn.GELU()
    def forward(self,x):
        return self.fc2(self.act(self.fc1(x)))

class HybridFreqGraphCrossAttention(nn.Module):
    def __init__(self, spec_d=128, graph_d=128, attn_d=128, heads=2):
        super().__init__()
        self.spec_proj = nn.Linear(spec_d, attn_d)
        self.graph_proj = nn.Linear(graph_d, attn_d)
        self.mha_spec = nn.MultiheadAttention(attn_d, heads, batch_first=True)
        self.mha_graph = nn.MultiheadAttention(attn_d, heads, batch_first=True)
        self.out_s = nn.Linear(attn_d, spec_d)
        self.out_g = nn.Linear(attn_d, graph_d)
    def forward(self, spec_tokens, graph_nodes):
        s = self.spec_proj(spec_tokens)
        g = self.graph_proj(graph_nodes)
        s_up,_ = self.mha_spec(s, g, g)
        g_up,_ = self.mha_graph(g, s, s)
        s_agg = s_up.mean(dim=1); g_agg = g_up.mean(dim=1)
        return self.out_s(s_agg), self.out_g(g_agg)

class TriAttentionFusion(nn.Module):
    def __init__(self, t_d=512, s_d=128, g_d=128, proj=128, fused=256):
        super().__init__()
        self.tp = nn.Linear(t_d, proj); self.sp = nn.Linear(s_d, proj); self.gp = nn.Linear(g_d, proj)
        self.logt = nn.Linear(proj,1); self.logs = nn.Linear(proj,1); self.logg = nn.Linear(proj,1)
        self.fusion = nn.Sequential(nn.Linear(proj, fused), nn.GELU(), nn.Dropout(0.2), nn.Linear(fused,fused), nn.BatchNorm1d(fused), nn.GELU())
    def forward(self,t,s,g):
        Tp=self.tp(t); Sp=self.sp(s); Gp=self.gp(g)
        logits = torch.cat([self.logt(Tp), self.logs(Sp), self.logg(Gp)], dim=1)
        weights = F.softmax(logits, dim=1)
        w_t,w_s,w_g = weights[:,0:1], weights[:,1:2], weights[:,2:3]
        fused = w_t*Tp + w_s*Sp + w_g*Gp
        fused = self.fusion(fused)
        return fused, weights

class SliceImportanceNet(nn.Module):
    def __init__(self, emb=256, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, hid), nn.GELU(), nn.Linear(hid,1))
    def forward(self, slice_embs):
        logits = self.net(slice_embs).squeeze(-1)
        return F.softmax(logits, dim=1)

# -------------------------
# NSMNet (same general architecture)
# -------------------------
class NSMNet(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=4,
                 dct_k=256, patch_grid=(4,4), spec_tokens=8, attn_dim=128, attn_heads=2,
                 use_mcs=True, use_hfg=True, use_asiw=True):
        super().__init__()
        self.spec_tokens = max(1, spec_tokens)
        self.dct_k = dct_k
        self.use_mcs = use_mcs
        self.use_hfg = use_hfg
        self.use_asiw = use_asiw
        # backbone
        self.backbone_name = backbone_name
        try:
            # try torchvision efficientnet v2 small first
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            feat_dummy = self.backbone.features(torch.zeros(1,3,IMG_SIZE,IMG_SIZE))
            feat_dim = feat_dummy.shape[1]
        except Exception:
            try:
                # fallback to timm
                self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
                feat_dim = self.backbone.num_features
            except Exception:
                # tiny fallback
                self.backbone = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
                feat_dim = 32

        self.texture_fc = nn.Sequential(nn.Linear(feat_dim,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2))

        self.spectral_net = SpectralNet(in_k=dct_k, out_d=128)
        self.patch_grid = patch_grid
        self.node_feat_dim = 36
        self.graph_enc = GCNEncoder(node_feat_dim=self.node_feat_dim, hidden=64, out_dim=128)
        self.mid_conv = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((patch_grid[0], patch_grid[1])))

        if self.use_mcs:
            self.mcs_mask = nn.Sequential(nn.Linear(128, dct_k), nn.Sigmoid())

        if self.use_hfg:
            self.hfg = HybridFreqGraphCrossAttention(spec_d=128, graph_d=128, attn_d=attn_dim, heads=attn_heads)

        self.trifuse = TriAttentionFusion(t_d=512, s_d=128, g_d=128, proj=128, fused=256)
        self.classifier = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

        if self.use_asiw:
            self.asw = SliceImportanceNet(emb=256, hid=64)

        token_group_len = max(1, dct_k // self.spec_tokens)
        self.spec_token_proj = nn.Linear(token_group_len, 128)
        self.node_proj = nn.Linear(self.node_feat_dim, 128)

    def forward_backbone_feat(self, x):
        if hasattr(self.backbone,"features"):
            feat_map = self.backbone.features(x)
            pooled = F.adaptive_avg_pool2d(feat_map, 1).view(x.size(0), -1)
            return pooled, feat_map
        else:
            try:
                pooled = self.backbone(x)
                return pooled, None
            except:
                pooled = F.adaptive_avg_pool2d(self.backbone(x), 1).view(x.size(0), -1)
                return pooled, None

    def compute_dct_batch(self, x_gray):
        B = x_gray.shape[0]; k = self.dct_k
        out=[]
        x_np = x_gray.detach().cpu().numpy()
        for i in range(B):
            arr = x_np[i,0]
            vec = compute_2d_dct_np(arr, k=k)
            out.append(vec)
        out = np.stack(out, axis=0).astype(np.float32)
        return torch.from_numpy(out).to(x_gray.device)

    def build_patch_graph(self, x):
        B,C,H,W = x.shape
        gx,gy = self.patch_grid
        ph, pw = H // gx, W // gy
        N = gx*gy
        mid = self.mid_conv(x)
        mid = mid.permute(0,2,3,1).reshape(B, N, 32)
        node_feats = []
        adjs = []
        for b in range(B):
            nodes=[]
            for i in range(gx):
                for j in range(gy):
                    ph0,ph1 = i*ph,(i+1)*ph
                    pw0,pw1 = j*pw,(j+1)*pw
                    patch = x[b:b+1,0,ph0:ph1,pw0:pw1]
                    mean = float(patch.mean().item()); std = float(patch.std().item())
                    cx=(i+0.5)/gx; cy=(j+0.5)/gy
                    local_feat = mid[b, i*gy + j].detach()
                    node = torch.cat([torch.tensor([mean,std,cx,cy], device=x.device), local_feat], dim=0)
                    nodes.append(node.cpu().numpy())
            nodes = np.stack(nodes, axis=0).astype(np.float32)
            node_feats.append(nodes)
            adj = np.zeros((N,N), dtype=np.float32)
            for i in range(gx):
                for j in range(gy):
                    idx = i*gy + j
                    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < gx and 0 <= nj < gy:
                            nidx = ni*gy + nj
                            adj[idx, nidx] = 1.0
            adj += np.eye(N, dtype=np.float32)
            adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-6)
            adjs.append(adj)
        node_feats = torch.from_numpy(np.stack(node_feats, axis=0)).to(x.device)
        adjs = torch.from_numpy(np.stack(adjs, axis=0)).to(x.device)
        return node_feats, adjs

    def forward(self, x, precomp_dct_vec=None):
        B = x.size(0)
        pooled_feat, feat_map = self.forward_backbone_feat(x)
        t_feat = self.texture_fc(pooled_feat)

        x_gray = ((x[:,0:1,:,:] + x[:,1:2,:,:] + x[:,2:3,:,:]) / 3.0 + 1.0) / 2.0
        if precomp_dct_vec is not None:
            dct_vec = precomp_dct_vec.to(x.device)
        else:
            dct_vec = self.compute_dct_batch(x_gray)

        node_feats, adj = self.build_patch_graph(x)
        graph_emb = self.graph_enc(node_feats, adj)

        if self.use_mcs:
            mask = self.mcs_mask(graph_emb)
            masked_dct = dct_vec * mask
        else:
            masked_dct = dct_vec
            mask = torch.ones_like(dct_vec)

        s_feat = self.spectral_net(masked_dct)

        if self.use_hfg:
            m = self.spec_tokens
            group_len = max(1, masked_dct.shape[1] // m)
            spec_tokens = masked_dct.unfold(1, group_len, group_len)
            if spec_tokens.dim() == 3:
                if spec_tokens.shape[-1] != self.spec_token_proj.in_features:
                    need = self.spec_token_proj.in_features - spec_tokens.shape[-1]
                    if need > 0:
                        pad = torch.zeros((spec_tokens.shape[0], spec_tokens.shape[1], need), device=x.device, dtype=spec_tokens.dtype)
                        spec_tokens = torch.cat([spec_tokens.to(x.device), pad], dim=2)
                spec_tokens = spec_tokens.to(x.device)
            else:
                st = masked_dct.unsqueeze(1)
                if st.shape[-1] != self.spec_token_proj.in_features:
                    need = self.spec_token_proj.in_features - st.shape[-1]
                    if need > 0:
                        pad = torch.zeros((st.shape[0], st.shape[1], need), device=x.device, dtype=st.dtype)
                        st = torch.cat([st.to(x.device), pad], dim=2)
                spec_tokens = st.to(x.device)

            spec_tokens_f = self.spec_token_proj(spec_tokens)  # B x m x 128
            node_proj_out = self.node_proj(node_feats)        # B x N x 128
            s_out, g_out = self.hfg(spec_tokens_f, node_proj_out)
            s_feat = s_feat + s_out
            graph_emb = graph_emb + g_out

        fused, tri_weights = self.trifuse(t_feat, s_feat, graph_emb)
        slice_emb = fused
        logits = self.classifier(slice_emb)
        return {"logits": logits, "slice_emb": slice_emb, "tri_weights": tri_weights, "graph_emb": graph_emb, "mask": mask}

# -------------------------
# Focal Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1-pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------------------------
# Robust loader (copy into file)
# -------------------------

def robust_load_state_dict(path, model, verbose=True):
    import torch
    if not os.path.exists(path):
        if verbose: print(f"robust_load: checkpoint {path} does not exist")
        return
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sd = ck
    elif isinstance(ck, dict) and 'state_dict' in ck:
        sd = ck['state_dict']
    else:
        sd = ck

    model_sd = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    mismatched = []
    new_sd = {}

    for k, v in sd.items():
        if k in model_sd:
            try:
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    new_sd[k] = v
                    loaded_keys.append(k)
                else:
                    mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            except Exception:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    if len(new_sd) > 0:
        model_sd.update(new_sd)
        model.load_state_dict(model_sd)
        if verbose:
            print(f"[robust_load] loaded {len(new_sd)} keys, skipped {len(skipped_keys)} unexpected keys, {len(mismatched)} shape-mismatched keys.")
    else:
        if verbose:
            print("[robust_load] No matching keys to load — attempting load_state_dict(..., strict=False)")
        try:
            model.load_state_dict(sd, strict=False)
            if verbose: print("[robust_load] load_state_dict(..., strict=False) succeeded.")
            return
        except Exception as e:
            if verbose: print("[robust_load] fallback strict=False also failed:", e)
            raise e

    if verbose and len(mismatched) > 0:
        print("=== MISMATCHED KEYS (key, ckpt_shape, model_shape) ===")
        for k,s_ck,s_mod in mismatched[:40]:
            print(k, s_ck, "->", s_mod)
        if len(mismatched) > 40:
            print("... (showing first 40 mismatches)")

    if verbose and len(skipped_keys) > 0:
        print("=== SKIPPED KEYS (unexpected in checkpoint; first 40) ===")
        for k in skipped_keys[:40]:
            print(k)
        if len(skipped_keys) > 40:
            print("... (showing first 40 skipped keys)")

# -------------------------
# Loss + metrics
# -------------------------
def patient_level_loss_and_metrics(model, batch_slice_embs, batch_logits, patient_labels, use_asiw, criterion_fn):
    device = batch_slice_embs.device
    P_, S_, D = batch_slice_embs.shape
    if use_asiw and model.use_asiw:
        weights = model.asw(batch_slice_embs)  # P x S
    else:
        weights = torch.ones((P_, S_), device=device) / float(S_)
    slice_probs = F.softmax(batch_logits, dim=-1)
    patient_probs = (weights.unsqueeze(-1) * slice_probs).sum(dim=1)
    ce = criterion_fn(patient_probs, patient_labels)
    cons = 0.0
    if S_ > 1:
        cons = F.mse_loss(batch_slice_embs[:, :-1, :], batch_slice_embs[:, 1:, :])
    total = ce + 0.1 * cons
    return total, {"ce": float(ce.detach().cpu().item()), "cons": float(cons.detach().cpu().item() if isinstance(cons, torch.Tensor) else cons)}

# -------------------------
# Training / Eval loops
# -------------------------

def print_gpu_mem(epoch_tag="epoch"):
    if not torch.cuda.is_available():
        print(f"[GPU] no CUDA available")
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved  = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"[GPU] {epoch_tag} | allocated: {allocated:.1f} MB | reserved: {reserved:.1f} MB | "
          f"max_alloc: {max_alloc:.1f} MB | max_reserved: {max_reserved:.1f} MB")
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def train_one_epoch_patientwise(model, patient_sampler, cfg, optimizer, scaler, epoch_idx, criterion_fn):
    model.train()
    running_loss = 0.0; steps = 0
    for batch_files, batch_pids, batch_labels in tqdm(patient_sampler, desc=f"Train E{epoch_idx}", leave=False):
        imgs=[]; precomp_vecs=[]
        for p in batch_files:
            img_t, dct_v = load_image_and_optional_dct(p, train_tf, PRECOMP_DCT)
            imgs.append(img_t); precomp_vecs.append(dct_v)
        imgs = torch.stack(imgs, dim=0).to(DEVICE)
        if any(v is not None for v in precomp_vecs):
            arrs=[]
            for v in precomp_vecs:
                if v is None: arrs.append(torch.zeros(cfg["dct_k"], dtype=torch.float32))
                else: arrs.append(v)
            precomp = torch.stack(arrs, dim=0).to(DEVICE)
        else:
            precomp = None

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=('cuda' if DEVICE=='cuda' else 'cpu'), enabled=(DEVICE=='cuda')):
            outs = model(imgs, precomp)
            logits = outs["logits"]; emb = outs["slice_emb"]
            PS = logits.size(0); P_batch = len(batch_pids); S_batch = PS // P_batch
            logits_ps = logits.view(P_batch, S_batch, -1)
            emb_ps = emb.view(P_batch, S_batch, -1)
            patient_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=DEVICE)
            total_loss, metrics = patient_level_loss_and_metrics(model, emb_ps, logits_ps, patient_labels_tensor, USE_ASIW, criterion_fn)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            total_loss.backward(); optimizer.step()
        running_loss += float(total_loss.item()); steps += 1
    return running_loss / (steps + 1e-12)

@torch.no_grad()
def eval_patientwise(model, df_val, cfg):
    model.eval()
    p2files={}; p2label={}
    for pid,g in df_val.groupby('pid'):
        p2files[pid] = g['path'].tolist(); p2label[pid]=g['label'].iloc[0]
    y_true=[]; y_pred=[]
    for pid, paths in tqdm(p2files.items(), desc="Eval patients", leave=False):
        if len(paths) > S:
            sel = random.sample(paths, S)
        else:
            sel = paths
        imgs=[]; precomp=[]
        for p in sel:
            img_t, dct_v = load_image_and_optional_dct(p, val_tf, PRECOMP_DCT)
            imgs.append(img_t); precomp.append(dct_v)
        imgs = torch.stack(imgs, dim=0).to(DEVICE)
        if any(v is not None for v in precomp):
            arrs=[]
            for v in precomp:
                if v is None: arrs.append(torch.zeros(cfg["dct_k"], dtype=torch.float32))
                else: arrs.append(v)
            precomp_vec = torch.stack(arrs, dim=0).to(DEVICE)
        else:
            precomp_vec = None
        outs = model(imgs, precomp_vec)
        logits = outs["logits"].cpu()
        if USE_ASIW and model.use_asiw:
            emb = outs["slice_emb"].cpu(); emb_ps = emb.unsqueeze(0)
            weights = model.asw(emb_ps.to(DEVICE)).squeeze(0).detach().cpu().numpy()
        else:
            weights = np.ones((logits.size(0),), dtype=np.float32) / float(logits.size(0))
        slice_probs = F.softmax(logits, dim=1).numpy()
        avgp = (weights[:,None] * slice_probs).sum(axis=0)
        pred = int(avgp.argmax())
        y_pred.append(pred); y_true.append(int(p2label[pid]))
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean() if len(y_true)>0 else 0.0
    return {"patient_acc": acc, "y_true": y_true, "y_pred": y_pred}

# -------------------------
# Grad-CAM & checkpoint helpers
# -------------------------
def run_gradcam_and_save(model, sample_paths, out_dir, n_show=8):
    if not HAS_GRADCAM:
        print("Skipping Grad-CAM: pytorch_grad_cam not installed.")
        return
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    target=None
    if hasattr(model.backbone, "features"):
        for layer in reversed(list(model.backbone.features.children())):
            if isinstance(layer, nn.Conv2d):
                target = layer; break
    if target is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                target = m; break
    try:
        # newer pytorch-grad-cam versions accept device argument instead of use_cuda
        try:
            cam = GradCAM(model=model, target_layers=[target], device=('cuda' if DEVICE=='cuda' else 'cpu'))
        except TypeError:
            cam = GradCAM(model=model, target_layers=[target], use_cuda=(DEVICE=='cuda'))
    except Exception as e:
        print("GradCAM init failed:", e); return
    for i, p in enumerate(sample_paths[:n_show]):
        img = Image.open(p).convert('RGB').resize((IMG_SIZE,IMG_SIZE))
        arr = np.array(img).astype(np.float32)/255.0
        input_t = val_tf(img).to(DEVICE)
        try:
            grayscale_cam = cam(input_tensor=input_t.unsqueeze(0))
            vis = show_cam_on_image(arr, grayscale_cam[0], use_rgb=True)
            plt.imsave(os.path.join(out_dir, f"gradcam_{i}.png"), vis)
        except Exception as e:
            print("GradCAM failed for", p, e)


def save_checkpoint(path, model, optimizer, scheduler, epoch, stage, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ck = {"epoch": epoch, "stage": stage, "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
          "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
          "best_val": best_val}
    torch.save(ck, path)
    try:
        weights_path = os.path.join(os.path.dirname(path), "best_stage2_weights.pth")
        torch.save(model.state_dict(), weights_path)
    except Exception as e:
        print("Warning: weights-only save failed:", e)
    # save metadata too
    try:
        meta = {"backbone_name": getattr(model,'backbone_name',None), "date": time.ctime(), "best_val": best_val}
        with open(os.path.join(os.path.dirname(path), "checkpoint_meta.json"), 'w') as mf:
            json.dump(meta, mf, indent=2)
    except Exception:
        pass

# -------------------------
# CV driver
# -------------------------
def run_cv(cfg):
    classes, cls2idx, patient_map, patient_label = build_patient_index(cfg["data_root"])
    print("Found classes:", classes)
    patient_ids = sorted(list(patient_map.keys()))
    print("Sample patient ids (first 20):", patient_ids[:20])

    pcounts = Counter([patient_label[pid] for pid in patient_ids])
    class_weights = [1.0 / float(pcounts[i]) if pcounts[i]>0 else 1.0 for i in range(len(classes))]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    focal_criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor if sum(class_weights)>0 else None)

    splits = stratified_patient_folds(patient_ids, patient_label, n_splits=cfg["num_folds"], seed=SEED)

    fold_results = {}
    for fold, (train_pids, val_pids) in enumerate(splits):
        print("\n==== Fold", fold, "====")
        print("train patient counts:", Counter([patient_label[pid] for pid in train_pids]))
        print("val patient counts:", Counter([patient_label[pid] for pid in val_pids]))

        def gather(pids):
            lst=[]
            for pid in pids:
                for path,lbl in patient_map[pid]:
                    lst.append((path,lbl,pid))
            return lst
        train_df = gather(train_pids); val_df = gather(val_pids); test_df = []
        df_train = pd.DataFrame(train_df, columns=['path','label','pid'])
        df_val = pd.DataFrame(val_df, columns=['path','label','pid'])

        missing_pids = [pid for pid in train_pids if pid not in patient_map]
        if len(missing_pids) > 0:
             print("WARNING: these train_pids are missing from patient_map and will be skipped:", missing_pids)

        sampler_map = {pid: patient_map[pid] for pid in train_pids if pid in patient_map}

        patient_sampler = BalancedClassPatientSampler(
           sampler_map,
           patient_label,
           patients_per_batch=P,
           slices_per_patient=S
         )
 
        model = NSMNet(backbone_name='efficientnet_b0', pretrained=True, num_classes=len(classes),
                       dct_k=256, patch_grid=(4,4), spec_tokens=8, attn_dim=128, attn_heads=2,
                       use_mcs=USE_MCSM, use_hfg=USE_HFG, use_asiw=USE_ASIW)
        model.to(DEVICE)

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2,PATIENCE//2))

        fold_dir = os.path.join(cfg["out_root"], f"fold_{fold}"); os.makedirs(fold_dir, exist_ok=True)
        ckpt_path = os.path.join(fold_dir, "checkpoint.pth"); best_stage2 = os.path.join(fold_dir, "best_stage2.pth")
        if PRECOMP_DCT:
            precompute_dct_for_dataset({pid: patient_map[pid] for pid in train_pids + val_pids + test_df}, k=256, img_size=IMG_SIZE)

        csv_path = os.path.join(fold_dir, 'train_log.csv')
        with open(csv_path, 'w', newline='') as cf:
            w = csv.writer(cf); w.writerow(['epoch','stage','train_loss','val_acc','macro_f1'])

        # Stage 1
        best_val = 0.0; early = 0
        scaler = torch.amp.GradScaler() if DEVICE=='cuda' else None

        for e in range(E1):
            tloss = train_one_epoch_patientwise(model, patient_sampler, cfg, opt, scaler, e, focal_criterion)
            val_stats = eval_patientwise(model, df_val, cfg)
            val_acc = val_stats["patient_acc"]
            print(f"[Fold{fold}][S1] Epoch {e} TrainLoss {tloss:.4f} ValAcc {val_acc:.4f}")
            print_gpu_mem(f"Fold{fold} Stage1 Epoch{e}")
            scheduler.step(1.0 - val_acc)
            if MAX_VAL_ACC is not None and val_acc >= MAX_VAL_ACC:
                early += 1
            else:
                if val_acc > best_val:
                    best_val = val_acc; torch.save(model.state_dict(), os.path.join(fold_dir, "best_stage1.pth")); early = 0
                else:
                    early += 1
            save_checkpoint(ckpt_path, model, opt, scheduler, e, stage=1, best_val=best_val)
            with open(csv_path, 'a', newline='') as cf:
                w = csv.writer(cf); w.writerow([e,1,tloss,val_acc,0.0])
            if early >= PATIENCE:
                print("Early stopping stage1"); break

        # Stage 2
        best1 = os.path.join(fold_dir, "best_stage1.pth")
        if os.path.exists(best1):
            try:
                model.load_state_dict(torch.load(best1, map_location=DEVICE))
            except Exception:
                # try robust partial load
                robust_load_state_dict(best1, model)
        if hasattr(model.backbone, "features"):
            children = list(model.backbone.features.children())
            n_unfreeze = max(1, min(6, len(children)//4))
            for layer in children[-n_unfreeze:]:
                for p in layer.parameters(): p.requires_grad = True
        for p in model.classifier.parameters(): p.requires_grad = True
        opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR2, weight_decay=1e-6)
        sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, factor=0.5, patience=max(2,PATIENCE//2))
        scaler2 = torch.amp.GradScaler() if DEVICE=='cuda' else None

        early2 = 0; best_val2 = best_val
        for e in range(E2):
            tloss = train_one_epoch_patientwise(model, patient_sampler, cfg, opt2, scaler2, e, focal_criterion)
            val_stats = eval_patientwise(model, df_val, cfg)
            val_acc = val_stats["patient_acc"]
            print(f"[Fold{fold}][S2] Epoch {e} TrainLoss {tloss:.4f} ValAcc {val_acc:.4f}")
            print_gpu_mem(f"Fold{fold} Stage2 Epoch{e}")
            sched2.step(1.0 - val_acc)
            if MAX_VAL_ACC is not None and val_acc >= MAX_VAL_ACC:
                early2 += 1
            else:
                if val_acc > best_val2:
                    best_val2 = val_acc
                    torch.save(model.state_dict(), best_stage2)
                    early2 = 0
                else:
                    early2 += 1
            save_checkpoint(ckpt_path, model, opt2, sched2, e, stage=2, best_val=best_val2)
            with open(csv_path, 'a', newline='') as cf:
                w = csv.writer(cf); w.writerow([e,2,tloss,val_acc,0.0])
            if early2 >= PATIENCE:
                print("Early stop stage2"); break

        # Evaluate best model
        best_model_path = best_stage2 if os.path.exists(best_stage2) else ckpt_path
        print("Loading best for eval:", best_model_path)
        model_eval = NSMNet(backbone_name='efficientnet_b0', pretrained=True, num_classes=len(classes),
                            dct_k=256, patch_grid=(4,4), spec_tokens=8, attn_dim=128, attn_heads=2,
                            use_mcs=USE_MCSM, use_hfg=USE_HFG, use_asiw=USE_ASIW)
        model_eval.to(DEVICE)
        try:
            robust_load_state_dict(best_model_path, model_eval)
        except Exception as e:
            print("Error loading best_model_path with robust loader:", e)
        test_df_pd = df_val
        test_stats = eval_patientwise(model_eval, test_df_pd, cfg)
        print(f"[Fold {fold}] Test patient acc: {test_stats['patient_acc']:.4f}")
        y_true = test_stats['y_true']; y_pred = test_stats['y_pred']
        if len(y_true)>0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111)
            cax = ax.matshow(cm, cmap='Blues')
            for (i,j),val in np.ndenumerate(cm):
                ax.text(j, i, int(val), ha='center', va='center')
            ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
            plt.title(f'Fold {fold} Confusion Matrix')
            plt.tight_layout(); plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png')); plt.close()
            with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as rf:
                rf.write(classification_report(y_true, y_pred, target_names=classes))

        sample_paths = [p for p,_,_ in val_df][:min(16,len(val_df))]
        run_gradcam_and_save(model_eval, sample_paths, os.path.join(fold_dir, "gradcam"), n_show=8)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump({"patient_acc": test_stats["patient_acc"]}, f, indent=2)
        fold_results[f"fold_{fold}"] = {"patient_acc": test_stats["patient_acc"]}

    accs = [v["patient_acc"] for v in fold_results.values()] if fold_results else []
    mean_acc = float(np.mean(accs)) if len(accs)>0 else 0.0
    std_acc = float(np.std(accs)) if len(accs)>0 else 0.0
    with open(os.path.join(cfg["out_root"], "cv_summary.json"), "w") as f:
        json.dump({"per_fold": fold_results, "mean": mean_acc, "std": std_acc}, f, indent=2)
    print("CV complete. mean acc:", mean_acc, "std:", std_acc)

# -------------------------
# Stratified helper and plotting
# -------------------------

def stratified_patient_folds(patient_ids, patient_label, n_splits=5, seed=42):
    rng = random.Random(seed)
    cls2p = defaultdict(list)
    for pid in patient_ids:
        cls2p[patient_label[pid]].append(pid)
    for cls in cls2p:
        rng.shuffle(cls2p[cls])
    folds = [ [] for _ in range(n_splits) ]
    for cls, pids in cls2p.items():
        for i, pid in enumerate(pids):
            folds[i % n_splits].append(pid)
    splits = []
    for k in range(n_splits):
        val = folds[k]
        train = []
        for j in range(n_splits):
            if j == k: continue
            train += folds[j]
        splits.append((train, val))
    return splits


def plot_fold_logs(out_root, save_plots=True):
    folds = [d for d in sorted(os.listdir(out_root)) if d.startswith('fold_')]
    for f in folds:
        csv_path = os.path.join(out_root, f, 'train_log.csv')
        if not os.path.exists(csv_path):
            print('No log for', f); continue
        df = pd.read_csv(csv_path)
        if df.empty: continue
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(df['epoch'], df['train_loss'], label='train_loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train_loss')
        ax2 = ax1.twinx()
        ax2.plot(df['epoch'], df['val_acc'], label='val_acc', color='tab:orange')
        ax2.set_ylabel('val_acc')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='upper right')
        plt.title(f + ' train loss / val acc')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(out_root, f, 'train_val_curve.png'))
        else:
            plt.show()
        plt.close(fig)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    config = {
        "data_root": DATA_ROOT,
        "out_root": OUT_ROOT,
        "num_folds": N_FOLDS,
        "batch_patients": P,
        "slices_per_patient": S,
        "img_size": IMG_SIZE,
        "epochs_stage1": E1,
        "epochs_stage2": E2,
        "dct_k": 256
    }
    print("Building patient index from", DATA_ROOT)
    classes, cls2idx, patient_map, patient_label = build_patient_index(DATA_ROOT)
    print(f"Found {len(classes)} classes, {len(patient_map)} patients")
    if PRECOMP_DCT:
        precompute_dct_for_dataset(patient_map, k=256, img_size=IMG_SIZE)
    if args.evaluate:
        for name in sorted(os.listdir(OUT_ROOT)):
            p = os.path.join(OUT_ROOT, name, "metrics.json")
            if os.path.exists(p):
                print(name, json.load(open(p)))
        sys.exit(0)
    run_cv(config)
