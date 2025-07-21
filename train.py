# train.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import CONFIG
from utils import evaluate, extract_tsne_feats
from dataloader import LncrnaDiseaseDataset
from model import JointModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(logger):
    """Run CV + final training with detailed logging and progress bars."""
    logger.info("==== Starting full training run ====")
    writer = SummaryWriter(log_dir=CONFIG["LOG_DIR"])
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Load data
    logger.info("Loading association and disease CSVs")
    assoc_df  = pd.read_csv(CONFIG["ASSOC_CSV"], index_col=0)
    disease_df= pd.read_csv(CONFIG["DISEASE_CSV"], index_col=0)
    assoc_df.index = assoc_df.index.astype(str).str.replace(":", "-")
    lncrna_ids = assoc_df.index.tolist()

    # Build dataset (initial hard-negative mining inside __init__)
    logger.info("Initializing dataset with neg_ratio=%d", CONFIG["NEG_RATIO"])
    full_dataset = LncrnaDiseaseDataset(
        assoc_df, disease_df, CONFIG["DATA_DIR"], lncrna_ids,
        transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ]),
        neg_ratio=CONFIG["NEG_RATIO"],
        seed=CONFIG["SEED"],
        crop_size=CONFIG["CGR_CROP_SIZE"],
        t=CONFIG["CGR_IUS_T"]
    )
    logger.info("Total pairs after filtering missing files: %d", len(full_dataset))

    # Split train/val vs test
    labels = [p[2] for p in full_dataset.pairs]
    tv_idx, test_idx = train_test_split(
        range(len(full_dataset)), test_size=0.2,
        stratify=labels, random_state=CONFIG["SEED"]
    )
    train_val_ds = Subset(full_dataset, tv_idx)
    test_ds      = Subset(full_dataset, test_idx)
    test_loader  = DataLoader(
        test_ds, batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS_VAL"],
        pin_memory=True
    )
    logger.info("Train+Val size: %d, Test size: %d", len(train_val_ds), len(test_ds))

    # Cross-validation
    skf = StratifiedKFold(
        n_splits=CONFIG["N_FOLDS"],
        shuffle=True,
        random_state=CONFIG["SEED"]
    )
    metrics_summary = []

    for fold, (tr_i, vl_i) in enumerate(
        skf.split(np.zeros(len(tv_idx)), [full_dataset.pairs[i][2] for i in tv_idx]),
        1
    ):
        logger.info("---- Fold %d/%d ----", fold, CONFIG["N_FOLDS"])

        # Build Subsets and loaders (pairs fixed)
        tr_ds = Subset(full_dataset, [tv_idx[i] for i in tr_i])
        vl_ds = Subset(full_dataset, [tv_idx[i] for i in vl_i])

        tr_loader = DataLoader(tr_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                               num_workers=CONFIG["NUM_WORKERS_TRAIN"], pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                               num_workers=CONFIG["NUM_WORKERS_VAL"], pin_memory=True)

        model     = JointModel(disease_df.shape[1]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

        best_auc, no_imp = 0.0, 0

        for epoch in range(1, CONFIG["EPOCHS"]+1):
            logger.info("[Fold %d] Starting epoch %d/%d", fold, epoch, CONFIG["EPOCHS"])
            model.train()
            running_loss = 0.0

            # Training loop with progress bar
            loop = tqdm(tr_loader, desc=f"Fold{fold} Epoch{epoch} Train", leave=False)
            for imgs, dfs, lbs in loop:
                imgs, dfs, lbs = imgs.to(device), dfs.to(device), lbs.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    logits = model(imgs, dfs)
                    loss = criterion(logits, lbs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                loop.set_postfix(train_loss=running_loss/((loop.n+1)))

            scheduler.step()
            avg_train = running_loss / len(tr_loader)
            logger.info("[Fold %d][Epoch %d] Avg Train Loss: %.4f", fold, epoch, avg_train)
            writer.add_scalar(f"Fold{fold}/Train/Loss", avg_train, epoch)

            # Validation
            val_metrics = evaluate(model, vl_loader, device)
            logger.info(
                "[Fold %d][Epoch %d] Val AUC=%.4f, AUPR=%.4f, ACC=%.4f, F1=%.4f",
                fold, epoch,
                val_metrics["AUC"],
                val_metrics["AUPR"],
                val_metrics["ACC"],
                val_metrics["F1"]
            )
            writer.add_scalar(f"Fold{fold}/Val/AUC", val_metrics["AUC"], epoch)

            # Early stopping & checkpointing
            if val_metrics["AUC"] > best_auc:
                best_auc = val_metrics["AUC"]
                no_imp = 0
                best_wts = model.state_dict()
                ckpt_path = os.path.join(CONFIG["RESULT_DIR"], f"fold{fold}_best.pth")
                torch.save(best_wts, ckpt_path)
                logger.info("[Fold %d][Epoch %d] New best AUC, saved: %s", fold, epoch, ckpt_path)
            else:
                no_imp += 1

            if epoch % CONFIG["CHECKPOINT_INTERVAL"] == 0:
                ep_path = os.path.join(CONFIG["RESULT_DIR"], f"fold{fold}_epoch{epoch}.pth")
                torch.save(model.state_dict(), ep_path)
                logger.info("[Fold %d][Epoch %d] Checkpoint saved: %s", fold, epoch, ep_path)

            if no_imp >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                logger.info("[Fold %d] Early stopping at epoch %d", fold, epoch)
                break

        # Load best and record final fold metrics
        model.load_state_dict(best_wts)
        fm = evaluate(model, vl_loader, device)
        metrics_summary.append({
            'Fold':    f'Fold{fold}',
            'MCC':     fm['MCC'],
            'F1-score':fm['F1'],
            'Precision':fm['Precision'],
            'Recall':   fm['Recall'],
            'ACC':      fm['ACC'],
            'AUPR':     fm['AUPR'],
            'AUC':      fm['AUC'],
        })
        logger.info("[Fold %d] Final Val AUC: %.4f", fold, fm['AUC'])

        # Save ROC & PR plots
        fpr, tpr, _ = roc_curve(fm['labels'], fm['probs'])
        plt.figure(); plt.plot(fpr, tpr)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Fold {fold} ROC')
        roc_path = os.path.join(CONFIG["RESULT_DIR"], f'fold{fold}_roc.png')
        plt.savefig(roc_path); plt.close()
        logger.info("[Fold %d] ROC curve saved: %s", fold, roc_path)

        prec, rec, _ = precision_recall_curve(fm['labels'], fm['probs'])
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Fold {fold} PR')
        pr_path = os.path.join(CONFIG["RESULT_DIR"], f'fold{fold}_pr.png')
        plt.savefig(pr_path); plt.close()
        logger.info("[Fold %d] PR curve saved: %s", fold, pr_path)

    # CV summary
    df = pd.DataFrame(metrics_summary).set_index('Fold')
    df.loc['Avg'] = df.mean()
    df.loc['Std'] = df.std()
    summary_path = os.path.join(CONFIG["RESULT_DIR"], 'metrics_summary.csv')
    df.to_csv(summary_path)
    logger.info("Cross-validation summary saved: %s", summary_path)
    print(df)

    # Final training on all train+val data
    logger.info("==== Final training on all train+val data ====")
    # pairs already fixed, just use train_val_ds
    full_loader = DataLoader(
        train_val_ds,
        batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
        num_workers=CONFIG["NUM_WORKERS_TRAIN"], pin_memory=True
    )
    final_model = JointModel(disease_df.shape[1]).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=CONFIG["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])
    train_losses, test_losses = [], []

    for epoch in range(1, CONFIG["EPOCHS"]+1):
        logger.info("[Final] Starting epoch %d/%d", epoch, CONFIG["EPOCHS"])
        final_model.train()
        run_loss = 0.0

        loop = tqdm(full_loader, desc=f"Final Epoch{epoch} Train", leave=False)
        for imgs, dfs, lbs in loop:
            imgs, dfs, lbs = imgs.to(device), dfs.to(device), lbs.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits = final_model(imgs, dfs)
                loss = torch.nn.BCEWithLogitsLoss()(logits, lbs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            loop.set_postfix(train_loss=run_loss/((loop.n+1)))

        scheduler.step()
        avg_run = run_loss / len(full_loader)
        train_losses.append(avg_run)
        logger.info("[Final][Epoch %d] Avg Train+Val Loss: %.4f", epoch, avg_run)

        # Test loss
        final_model.eval()
        run_test = 0.0
        for imgs_t, dfs_t, lbs_t in tqdm(test_loader, desc="Final Test Eval", leave=False):
            imgs_t, dfs_t, lbs_t = imgs_t.to(device), dfs_t.to(device), lbs_t.to(device)
            with torch.no_grad(), autocast(device_type=device.type):
                logits_t = final_model(imgs_t, dfs_t)
                run_test += torch.nn.BCEWithLogitsLoss()(logits_t, lbs_t).item()
        avg_test = run_test / len(test_loader)
        test_losses.append(avg_test)
        logger.info("[Final][Epoch %d] Test Loss: %.4f", epoch, avg_test)

    # Final loss curves
    plt.figure()
    plt.plot(train_losses, label='Train+Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Final Training Loss Curves')
    plt.legend()
    flc_path = os.path.join(CONFIG["RESULT_DIR"], 'final_loss_curves.png')
    plt.savefig(flc_path); plt.close()
    logger.info("Final loss curves saved: %s", flc_path)

    # Final evaluation & plots
    tm = evaluate(final_model, test_loader, device)
    logger.info("[Final] Test metrics: AUC=%.4f, AUPR=%.4f, ACC=%.4f, F1=%.4f",
                tm['AUC'], tm['AUPR'], tm['ACC'], tm['F1'])

    fpr_t, tpr_t, _ = roc_curve(tm['labels'], tm['probs'])
    plt.figure(); plt.plot(fpr_t, tpr_t)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Test ROC')
    troc_path = os.path.join(CONFIG["RESULT_DIR"], 'test_roc.png')
    plt.savefig(troc_path); plt.close()
    logger.info("Test ROC saved: %s", troc_path)

    prec_t, rec_t, _ = precision_recall_curve(tm['labels'], tm['probs'])
    plt.figure(); plt.plot(rec_t, prec_t)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Test PR')
    tpr_path = os.path.join(CONFIG["RESULT_DIR"], 'test_pr.png')
    plt.savefig(tpr_path); plt.close()
    logger.info("Test PR saved: %s", tpr_path)

    cm = confusion_matrix(tm['labels'], [1 if p>=0.5 else 0 for p in tm['probs']])
    plt.figure(); plt.imshow(cm, interpolation='nearest')
    plt.title('Test Confusion Matrix'); plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['Neg','Pos']); plt.yticks(ticks, ['Neg','Pos'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    tcm_path = os.path.join(CONFIG["RESULT_DIR"], 'test_confusion_matrix.png')
    plt.savefig(tcm_path); plt.close()
    logger.info("Test confusion matrix saved: %s", tcm_path)

    conv, digit, labs = extract_tsne_feats(final_model, test_loader, device)
    tsne_conv  = TSNE(n_components=2, random_state=CONFIG['SEED']).fit_transform(conv)
    tsne_digit = TSNE(n_components=2, random_state=CONFIG['SEED']).fit_transform(digit)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(tsne_conv[labs==1,0], tsne_conv[labs==1,1], c='r', label='Pos', s=5)
    plt.scatter(tsne_conv[labs==0,0], tsne_conv[labs==0,1], c='b', label='Neg', s=5)
    plt.title('Conv Features t-SNE'); plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(tsne_digit[labs==1,0], tsne_digit[labs==1,1], c='r', label='Pos', s=5)
    plt.scatter(tsne_digit[labs==0,0], tsne_digit[labs==0,1], c='b', label='Neg', s=5)
    plt.title('DigitCaps Features t-SNE'); plt.legend()
    tsne_path = os.path.join(CONFIG["RESULT_DIR"], 'tsne_plots.png')
    plt.savefig(tsne_path); plt.close()
    logger.info("t-SNE plots saved: %s", tsne_path)
