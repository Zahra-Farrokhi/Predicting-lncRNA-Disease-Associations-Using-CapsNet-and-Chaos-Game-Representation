# dataloader.py
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import CONFIG
from utils import cgr_ius_crop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LncrnaDiseaseDataset(Dataset):
    def __init__(self, assoc_df, disease_gene_df, img_dir, lncrna_ids,
                 transform=None, neg_ratio=1, seed=42, crop_size=256, t=2):
        """Dataset with CGR-cropped images and negative sampling."""
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
        self.pos_pairs = [(i, j) for i, j in np.argwhere(self.assoc == 1)]
        self.neg_pool = [(i, j)
                         for i in range(self.assoc.shape[0])
                         for j in np.where(self.assoc[i] == 0)[0]]

        # generate initial random negatives, build self.pairs
        self._mine_hard_negatives(model=None)

        # cache for cropped images
        self._crop_cache = {}

    def _mine_hard_negatives(self, model=None, top_k=None):
        """
        Update self.neg_pairs and self.pairs:
        - If model is None: random sample neg_ratio * |pos| negatives
        - Else: choose top_k hardest negatives by model score
        """
        if model is None:
            idx = self.rng.choice(
                len(self.neg_pool),
                size=self.neg_ratio * len(self.pos_pairs),
                replace=False
            )
            self.neg_pairs = [self.neg_pool[i] for i in idx]
        else:
            # compute scores on all negatives
            sampler = WeightedRandomSampler(
                weights=[1.0] * len(self.neg_pool),
                num_samples=len(self.neg_pool),
                replacement=False
            )
            loader = DataLoader(
                self, batch_size=CONFIG["BATCH_SIZE"],
                sampler=sampler,
                num_workers=CONFIG["NUM_WORKERS_VAL"]
            )
            scores = []
            for imgs, dfs, _ in loader:
                imgs, dfs = imgs.to(device), dfs.to(device)
                with torch.no_grad(), torch.cuda.amp.autocast(device_type=device.type):
                    logits = model(imgs, dfs)
                scores.extend(torch.sigmoid(logits).cpu().numpy())
            top_k = top_k or (self.neg_ratio * len(self.pos_pairs))
            idxs = np.argsort(scores)[-top_k:]
            self.neg_pairs = [self.neg_pool[i] for i in idxs]

        # build combined list
        combined = [(i, j, 1) for (i, j) in self.pos_pairs] + \
                   [(i, j, 0) for (i, j) in self.neg_pairs]
        random.shuffle(combined)

        # **Filter out any pairs whose image file is missing**
        self.pairs = [
            (i, j, lbl) for (i, j, lbl) in combined
            if os.path.isfile(os.path.join(self.img_dir, f"{self.lncrna_ids[i]}.png"))
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        lncrna_id = self.lncrna_ids[i]
        img_path = os.path.join(self.img_dir, f"{lncrna_id}.png")

        # cache CGR-cropped images
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
        lbl   = torch.tensor(label, dtype=torch.float32)
        return img, d_feat, lbl
