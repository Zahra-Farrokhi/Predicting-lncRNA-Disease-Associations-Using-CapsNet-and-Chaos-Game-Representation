# utils.py
import os
import random
import datetime
import logging

import numpy as np
import torch
from torch.amp import autocast  # updated import
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
    accuracy_score, matthews_corrcoef, f1_score
)
from sklearn.manifold import TSNE

def seed_everything(seed=42):
    """Seed RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(cfg):
    """Configure console + file logger."""
    level = getattr(logging, "INFO", logging.INFO)
    logger = logging.getLogger("model")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # console handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # file handler
    log_file = os.path.join(cfg["LOG_DIR"],
                            f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Logging to file: %s", log_file)
    return logger

def squash(tensor, dim=-1):
    """Capsule squash activation."""
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

def cgr_ius_crop(img: Image.Image, crop_size: int = 256, t: int = 2) -> Image.Image:
    """CGR-based crop & zoom."""
    factor = 4 ** t
    w, h = img.size
    raw_crop_w, raw_crop_h = w // factor, h // factor
    img_cropped = img.crop((0, 0, raw_crop_w, raw_crop_h))
    img_zoomed = img_cropped.resize((crop_size, crop_size), Image.BICUBIC)
    return img_zoomed

def evaluate(model, dataloader, device):
    """Compute all classification metrics on dataloader."""
    model.eval()
    all_probs, all_labels = [], []
    # choose device_type for autocast
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for imgs, dfs, lbs in dataloader:
            imgs, dfs = imgs.to(device), dfs.to(device)
            # updated autocast usage
            with autocast(device_type=dev_type):
                logits = model(imgs, dfs)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbs.numpy())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
        "AUC": roc_auc_score(all_labels, all_probs),
        "AUPR": average_precision_score(all_labels, all_probs),
        "Precision": precision_score(all_labels, preds),
        "Recall": recall_score(all_labels, preds),
        "ACC": accuracy_score(all_labels, preds),
        "MCC": matthews_corrcoef(all_labels, preds),
        "F1": f1_score(all_labels, preds),
    }

def extract_tsne_feats(model, dataloader, device):
    """Extract conv & capsule features for t-SNE visualization."""
    model.eval()
    conv_list, digit_list, label_list = [], [], []
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for imgs, _, lbs in dataloader:
            imgs = imgs.to(device)
            with autocast(device_type=dev_type):
                x = model.img_net.features(imgs)
            conv_feat = x.view(x.size(0), x.size(1), -1).mean(-1).cpu().numpy()
            u = model.img_net.primary(x)
            v = model.img_net.digit(u).cpu().numpy()
            conv_list.append(conv_feat)
            digit_list.append(v.reshape(v.shape[0], -1))
            label_list.append(lbs.numpy())
    return (
        np.concatenate(conv_list, axis=0),
        np.concatenate(digit_list, axis=0),
        np.concatenate(label_list, axis=0),
    )
