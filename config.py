# config.py
import os
import platform

CONFIG = dict(
    DATA_DIR=r"cgr_images22",
    ASSOC_CSV=r"association-lncrna_disease_matrix_binary.csv",
    DISEASE_CSV=r"binary_feature_vector_gene_disease.csv",
    RESULT_DIR=r"RESULT5",
    LOG_DIR=r"logs",
    NEG_RATIO=1,
    BATCH_SIZE=16,
    EPOCHS=1,
    LR=1e-4,
    N_FOLDS=2,
    NUM_WORKERS_TRAIN=8,
    NUM_WORKERS_VAL=4,
    SEED=42,
    CGR_CROP_SIZE=128,
    CGR_IUS_T=2,
    EARLY_STOPPING_PATIENCE=5,
    CHECKPOINT_INTERVAL=3,
)

# ensure output dirs exist
os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

# On Windows, multiprocessing in DataLoader can hit paging‐file limits.
# Force single‐worker to avoid WinError 1455.
if platform.system() == "Windows":
    CONFIG["NUM_WORKERS_TRAIN"] = 0
    CONFIG["NUM_WORKERS_VAL"]   = 0
