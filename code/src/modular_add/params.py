import torch
import json
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Param:
    MODEL: str = "transformer"
    OPTIM: str = "adam"
    MODEL_PATH: str = "../model/transformer.pth"
    FIGURE_SAVE_PATH: str = "../figure/"
    MODULUS: int = 47

    # Default Hyperparameters
    SEED: int = 0
    EPOCH_NUM: int = 50000
    LR: float = 0.001
    BATCH_SIZE: int = 256
    TEST_ALPHA: float = 0.4
    D_MODEL: int = 16
    N_HEAD: int = 1
    DIM_FEEDFORWARD: int = 32
    N_LAYERS: int = 4
    MAX_SEQ_LENGTH: int = 8
    DROPOUT: float = 0.
    WEIGHT_DECAY: float = 0.
    STEP_LR_STEP_SIZE: int = 100
    STEP_LR_GAMMA: float = 0.98
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 100


def load_params(path=None):
    if path is None:
        return
    with open(path) as f:
        params = json.load(f)
        
    for key, value in params.items():
        setattr(Param, key.upper(), value)
    print("Loaded params from:", path)
