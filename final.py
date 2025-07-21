import os
import random
import datetime
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
    accuracy_score, matthews_corrcoef, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE

from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Configuration
# ---------------------------
CONFIG = dict(
    DATA_DIR=r"cgr_images22",
    ASSOC_CSV=r"association-lncrna_disease_matrix_binary.csv",
    DISEASE_CSV=r"binary_feature_vector_gene_disease.csv",
    RESULT_DIR=r"RESULT5",
    LOG_DIR=r"logs",
    NEG_RATIO=1,
    BATCH_SIZE=128,
    EPOCHS=2,
    LR=1e-4,
    N_FOLDS=5,
    NUM_WORKERS_TRAIN=8,
    NUM_WORKERS_VAL=4,
    SEED=42,
    CGR_CROP_SIZE=128,
    CGR_IUS_T=2,
    EARLY_STOPPING_PATIENCE=5,     # lower for faster demo
    CHECKPOINT_INTERVAL=3,
)

# make dirs
os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("the device is being used is : ",device)

# ---------------------------
# Utilities
# ---------------------------
def seed_everything(seed=42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(cfg):
    """Configure root logger to file and stream."""
    level = getattr(logging, "INFO", logging.INFO)
    logger = logging.getLogger("model")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # file
    log_file = os.path.join(cfg["LOG_DIR"], f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Logging to file: %s", log_file)
    return logger

def squash(tensor, dim=-1):
    """Squash activation for capsule networks."""
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

def cgr_ius_crop(img: Image.Image, crop_size: int = 256, t: int = 2) -> Image.Image:
    """CGR-based crop & zoom, cached per-file for speed."""
    factor = 4 ** t
    w, h = img.size
    raw_crop_w, raw_crop_h = w // factor, h // factor
    img_cropped = img.crop((0, 0, raw_crop_w, raw_crop_h))
    img_zoomed = img_cropped.resize((crop_size, crop_size), Image.BICUBIC)
    return img_zoomed

# ---------------------------
# Dataset with caching & hard-negative mining
# ---------------------------
class LncrnaDiseaseDataset(Dataset):
    def __init__(self, assoc_df, disease_gene_df, img_dir, lncrna_ids,
                 transform=None, neg_ratio=1, seed=42, crop_size=256, t=2):
        """
        assoc_df: lncRNA x disease binary matrix DataFrame
        disease_gene_df: disease feature DataFrame
        img_dir: folder of PNG images named <lncrna_id>.png
        """
        self.assoc = assoc_df.values
        self.disease_feats = disease_gene_df.values
        self.img_dir = img_dir
        self.lncrna_ids = lncrna_ids
        self.transform = transform
        self.neg_ratio = neg_ratio
        self.crop_size = crop_size
        self.t = t
        self.rng = np.random.default_rng(seed)
        # prepare positive and full negative pools
        self.pos_pairs = [(i,j) for i,j in np.argwhere(self.assoc==1)]
        self.neg_pool = [(i,j) for i in range(self.assoc.shape[0])
                         for j in np.where(self.assoc[i]==0)[0]]
        self._mine_hard_negatives()  # initial random negatives
        # cache for cropped images
        self._crop_cache = {}

    def _mine_hard_negatives(self, model=None, top_k=None):
        """
        Update self.neg_pairs:
        - if model is None: random sample neg_ratio * |pos|
        - else: choose top_k hardest negatives by model score
        """
        if model is None:
            # random negatives
            neg = self.rng.choice(len(self.neg_pool),
                                  size=self.neg_ratio*len(self.pos_pairs),
                                  replace=False)
            self.neg_pairs = [self.neg_pool[i] for i in neg]
        else:
            # compute logits on all negatives
            loader = DataLoader(self, batch_size=CONFIG["BATCH_SIZE"],
                                sampler=WeightedRandomSampler(weights=[1. if (i,j) in self.neg_pool else 0.
                                    for (i,j,_) in self.pairs], num_samples=len(self.neg_pool)),
                                num_workers=CONFIG["NUM_WORKERS_VAL"])
            scores = []
            for imgs, dfs, _ in loader:
                imgs, dfs = imgs.to(device), dfs.to(device)
                with torch.no_grad(), autocast():
                    logits = model(imgs, dfs)
                    scores.extend(torch.sigmoid(logits).cpu().numpy())
            # pick top_k hardest negatives
            top_k = top_k or self.neg_ratio*len(self.pos_pairs)
            idxs = np.argsort(scores)[-top_k:]
            self.neg_pairs = [self.neg_pool[i] for i in idxs]
        # build final labeled list
        self.pairs = [(i, j, 1) for (i,j) in self.pos_pairs] + [(i,j,0) for (i,j) in self.neg_pairs]
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        lncrna_id = self.lncrna_ids[i]
        img_path = os.path.join(self.img_dir, f"{lncrna_id}.png")
        # use cache to avoid re-cropping
        if img_path not in self._crop_cache:
            img = Image.open(img_path).convert("L")
            cropped = cgr_ius_crop(img, crop_size=self.crop_size, t=self.t)
            self._crop_cache[img_path] = cropped
        else:
            cropped = self._crop_cache[img_path]
        img = cropped
        if self.transform:
            img = self.transform(img)
        d_feat = torch.tensor(self.disease_feats[j], dtype=torch.float32)
        lbl = torch.tensor(label, dtype=torch.float32)
        return img, d_feat, lbl

# ---------------------------
# Model Definitions
# ---------------------------
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2,-1))*self.scale, dim=-1)
        return torch.matmul(attn, V)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_ch, out_caps, cap_dim, k, s):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_caps*cap_dim, k, s)
        self.out_caps, self.cap_dim = out_caps, cap_dim

    def forward(self, x):
        u = self.conv(x)
        B,C,H,W = u.size()
        u = u.view(B, self.out_caps, self.cap_dim, H*W).permute(0,1,3,2)
        u = u.contiguous().view(B, -1, self.cap_dim)
        return squash(u, dim=-1)

class DigitCapsules(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, iters=5):
        super().__init__()
        self.out_caps, self.out_dim, self.iters = out_caps, out_dim, iters
        self.W = None

    def forward(self, x):
        B,in_caps,in_dim = x.size()
        if self.W is None or self.W.size(1)!=in_caps:
            self.W = nn.Parameter(torch.randn(1, in_caps, self.out_caps, self.out_dim, in_dim, device=x.device))
        x = x.view(B,in_caps,in_dim,1).unsqueeze(2)
        W = self.W.expand(B,-1,-1,-1,-1)
        u_hat = torch.matmul(W, x)
        b = torch.zeros(B,in_caps,self.out_caps,1,1, device=x.device)
        for r in range(self.iters):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = squash(s, dim=-2)
            if r<self.iters-1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)
        v = v.squeeze(1).squeeze(-1)
        return v

class ImageCapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.primary = PrimaryCapsules(256, 8, 32, 9, 2)
        self.digit   = DigitCapsules(1152,32,16,16)

    def forward(self, x):
        x = self.features(x)
        u = self.primary(x)
        v = self.digit(u)
        return v.norm(dim=-1)

class DiseaseNet(nn.Module):
    def __init__(self, ng):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ng,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU()
        )
    def forward(self, x):
        return self.mlp(x)

class JointModel(nn.Module):
    def __init__(self, ng):
        super().__init__()
        self.img_net = ImageCapsNet()
        self.dis_net = DiseaseNet(ng)
        self.attn    = Attention(80)
        self.fusion  = nn.Sequential(
            nn.Linear(80,128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,1)
        )
    def forward(self, i, df):
        img_feat = self.img_net(i)
        dis_feat = self.dis_net(df)
        combined = torch.cat([img_feat,dis_feat],1).unsqueeze(1)
        attended = self.attn(combined).squeeze(1)
        return self.fusion(attended).squeeze(1)

# ---------------------------
# Evaluation Helpers
# ---------------------------
def evaluate(model, dataloader, device):
    """Compute metrics on dataloader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, dfs, lbs in dataloader:
            imgs, dfs = imgs.to(device), dfs.to(device)
            with autocast():
                logits = model(imgs, dfs)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbs.numpy())
    preds = [1 if p>=0.5 else 0 for p in all_probs]
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
    """Extract conv & capsule features for t-SNE."""
    model.eval()
    conv_list, digit_list, label_list = [], [], []
    with torch.no_grad():
        for imgs, _, lbs in dataloader:
            imgs = imgs.to(device)
            with autocast():
                x = model.img_net.features(imgs)
            conv_feat = x.view(x.size(0), x.size(1), -1).mean(-1).cpu().numpy()
            u = model.img_net.primary(x)
            v = model.img_net.digit(u).cpu().numpy()
            conv_list.append(conv_feat)
            digit_list.append(v.reshape(v.shape[0], -1))
            label_list.append(lbs.numpy())
    return (np.concatenate(conv_list,0),
            np.concatenate(digit_list,0),
            np.concatenate(label_list,0))

# ---------------------------
# Main Training & CV
# ---------------------------
def main():
    seed_everything(CONFIG["SEED"])
    logger = setup_logger(CONFIG)
    writer = SummaryWriter(log_dir=CONFIG["LOG_DIR"])  # TensorBoard

    # Load data
    assoc_df = pd.read_csv(CONFIG["ASSOC_CSV"], index_col=0)
    disease_df = pd.read_csv(CONFIG["DISEASE_CSV"], index_col=0)
    assoc_df.index = assoc_df.index.astype(str).str.replace(":", "-")
    lncrna_ids = assoc_df.index.tolist()

    full_dataset = LncrnaDiseaseDataset(
        assoc_df, disease_df, CONFIG["DATA_DIR"], lncrna_ids,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ]),
        neg_ratio=CONFIG["NEG_RATIO"], seed=CONFIG["SEED"],
        crop_size=CONFIG["CGR_CROP_SIZE"], t=CONFIG["CGR_IUS_T"]
    )

    labels = [p[2] for p in full_dataset.pairs]
    train_val_idx, test_idx = train_test_split(
        range(len(full_dataset)), test_size=0.2,
        stratify=labels, random_state=CONFIG["SEED"]
    )
    train_val_dataset = Subset(full_dataset, train_val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"],
                             shuffle=False, num_workers=CONFIG["NUM_WORKERS_VAL"],
                             pin_memory=True)

    skf = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])

    metrics_summary = []
    scaler = GradScaler()  # for mixed precision across CV

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(train_val_idx)),
                  [full_dataset.pairs[i][2] for i in train_val_idx]), 1):

        # prepare per-fold dataset references
        full_dataset._mine_hard_negatives(model=None)  # random initial negs
        tr_ds = Subset(full_dataset, [train_val_idx[i] for i in train_idx])
        vl_ds = Subset(full_dataset, [train_val_idx[i] for i in val_idx])
        tr_loader = DataLoader(tr_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                               num_workers=CONFIG["NUM_WORKERS_TRAIN"], pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                               num_workers=CONFIG["NUM_WORKERS_VAL"], pin_memory=True)

        model = JointModel(disease_df.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

        best_val_auc, epochs_no_improve = 0, 0

        for epoch in range(1, CONFIG["EPOCHS"]+1):
            model.train()
            running_loss = 0.0

            # dynamic hard-negative mining at epoch start
            if epoch > 1:
                full_dataset._mine_hard_negatives(model=model)

            for imgs, dfs, lbs in tr_loader:
                imgs, dfs, lbs = imgs.to(device), dfs.to(device), lbs.to(device)
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs, dfs)
                    loss = criterion(logits, lbs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            scheduler.step()

            avg_train_loss = running_loss / len(tr_loader)
            writer.add_scalar(f"Fold{fold}/Train/Loss", avg_train_loss, epoch)

            # Validation
            val_metrics = evaluate(model, vl_loader, device)
            writer.add_scalar(f"Fold{fold}/Val/AUC", val_metrics["AUC"], epoch)
            writer.add_scalar(f"Fold{fold}/Val/Loss", 1 - val_metrics["AUC"], epoch)  # proxy

            # Early stopping & checkpointing
            if val_metrics["AUC"] > best_val_auc:
                best_val_auc = val_metrics["AUC"]
                epochs_no_improve = 0
                best_model_wts = model.state_dict()
                # save best
                torch.save(best_model_wts,
                           os.path.join(CONFIG["RESULT_DIR"], f"fold{fold}_best.pth"))
            else:
                epochs_no_improve += 1

            if epoch % CONFIG["CHECKPOINT_INTERVAL"] == 0:
                torch.save(model.state_dict(),
                           os.path.join(CONFIG["RESULT_DIR"], f"fold{fold}_epoch{epoch}.pth"))

            if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                logger.info(f"Fold {fold} early stopping at epoch {epoch}")
                break

        # load best and evaluate
        model.load_state_dict(best_model_wts)
        final_metrics = evaluate(model, vl_loader, device)
        metrics_summary.append({
            'Fold': f'Fold{fold}',
            'MCC': final_metrics['MCC'],
            'F1-score': final_metrics['F1'],
            'Precision': final_metrics['Precision'],
            'Recall': final_metrics['Recall'],
            'ACC': final_metrics['ACC'],
            'AUPR': final_metrics['AUPR'],
            'AUC': final_metrics['AUC'],
        })

        # save ROC & PR plots
        fpr, tpr, _ = roc_curve(final_metrics['labels'], final_metrics['probs'])
        prec, rec, _ = precision_recall_curve(final_metrics['labels'], final_metrics['probs'])
        plt.figure()
        plt.plot(fpr, tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Fold {fold} ROC')
        plt.savefig(os.path.join(CONFIG["RESULT_DIR"], f'fold{fold}_roc.png')); plt.close()
        plt.figure()
        plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Fold {fold} PR')
        plt.savefig(os.path.join(CONFIG["RESULT_DIR"], f'fold{fold}_pr.png')); plt.close()

    # summary CSV
    df_sum = pd.DataFrame(metrics_summary).set_index('Fold')
    df_sum.loc['Avg'] = df_sum.mean()
    df_sum.loc['Std'] = df_sum.std()
    df_sum.to_csv(os.path.join(CONFIG["RESULT_DIR"], 'metrics_summary.csv'))
    print(df_sum)

    # Final training on all train+val
    full_dataset._mine_hard_negatives(model=None)
    full_loader = DataLoader(train_val_dataset, batch_size=CONFIG["BATCH_SIZE"],
                             shuffle=True, num_workers=CONFIG["NUM_WORKERS_TRAIN"], pin_memory=True)
    final_model = JointModel(disease_df.shape[1]).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=CONFIG["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])
    train_final_loss, test_final_loss = [], []

    for epoch in range(1, CONFIG["EPOCHS"]+1):
        final_model.train()
        run_loss = 0
        for imgs, dfs, lbs in full_loader:
            imgs, dfs, lbs = imgs.to(device), dfs.to(device), lbs.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = final_model(imgs, dfs)
                loss = nn.BCEWithLogitsLoss()(logits, lbs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        scheduler.step()
        train_final_loss.append(run_loss / len(full_loader))

        # test loss
        final_model.eval()
        run_test_loss = 0
        with torch.no_grad():
            for imgs_t, dfs_t, lbs_t in test_loader:
                imgs_t, dfs_t, lbs_t = imgs_t.to(device), dfs_t.to(device), lbs_t.to(device)
                with autocast():
                    logits_t = final_model(imgs_t, dfs_t)
                    run_test_loss += nn.BCEWithLogitsLoss()(logits_t, lbs_t).item()
        test_final_loss.append(run_test_loss / len(test_loader))

    # plot final loss curves
    plt.figure()
    plt.plot(train_final_loss, label='Train+Val Loss')
    plt.plot(test_final_loss, label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Final Training Loss Curves')
    plt.legend(); plt.savefig(os.path.join(CONFIG["RESULT_DIR"], 'final_loss_curves.png')); plt.close()

    # test evaluation & plots
    test_metrics = evaluate(final_model, test_loader, device)
    fpr_t, tpr_t, _ = roc_curve(test_metrics['labels'], test_metrics['probs'])
    prec_t, rec_t, _ = precision_recall_curve(test_metrics['labels'], test_metrics['probs'])
    plt.figure(); plt.plot(fpr_t, tpr_t); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Test ROC')
    plt.savefig(os.path.join(CONFIG["RESULT_DIR"], 'test_roc.png')); plt.close()
    plt.figure(); plt.plot(rec_t, prec_t); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Test PR')
    plt.savefig(os.path.join(CONFIG["RESULT_DIR"], 'test_pr.png')); plt.close()

    cm = confusion_matrix(test_metrics['labels'], [1 if p>=0.5 else 0 for p in test_metrics['probs']])
    plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title('Test Confusion Matrix')
    plt.colorbar(); ticks = np.arange(2); plt.xticks(ticks, ['Neg','Pos']); plt.yticks(ticks, ['Neg','Pos'])
    for i in range(2):
        for j in range(2): plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.xlabel('Pred'); plt.ylabel('True'); plt.savefig(os.path.join(CONFIG["RESULT_DIR"], 'test_confusion_matrix.png')); plt.close()

    conv_feats, digit_feats, labels_tsne = extract_tsne_feats(final_model, test_loader, device)
    tsne_conv = TSNE(n_components=2, random_state=CONFIG['SEED']).fit_transform(conv_feats)
    tsne_digit = TSNE(n_components=2, random_state=CONFIG['SEED']).fit_transform(digit_feats)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(tsne_conv[labels_tsne==1,0], tsne_conv[labels_tsne==1,1], c='r', label='Pos', s=5)
    plt.scatter(tsne_conv[labels_tsne==0,0], tsne_conv[labels_tsne==0,1], c='b', label='Neg', s=5)
    plt.title('Conv Features t-SNE'); plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(tsne_digit[labels_tsne==1,0], tsne_digit[labels_tsne==1,1], c='r', label='Pos', s=5)
    plt.scatter(tsne_digit[labels_tsne==0,0], tsne_digit[labels_tsne==0,1], c='b', label='Neg', s=5)
    plt.title('DigitCaps Features t-SNE'); plt.legend()
    plt.savefig(os.path.join(CONFIG["RESULT_DIR"], 'tsne_plots.png')); plt.close()

if __name__ == '__main__':
    main()
