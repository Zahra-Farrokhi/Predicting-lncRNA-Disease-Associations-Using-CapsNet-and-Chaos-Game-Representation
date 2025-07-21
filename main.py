# main.py
from config import CONFIG
from utils import seed_everything, setup_logger
from train import run

def main():
    # seed + logger
    seed_everything(CONFIG["SEED"])
    logger = setup_logger(CONFIG)
    # kick off CV + final training
    run(logger)

if __name__ == "__main__":
    main()
