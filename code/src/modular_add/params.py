import torch
import json
import math
import re
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Param:
    ### Environment ###
    MODEL: str = "transformer"
    OPTIMIZER: str = "adam"
    MODEL_PATH: str = "../model/transformer.pth"
    FIGURE_SAVE_PATH: str = "../fig/"
    MODULUS: int = 97
    SEED: int = 0

    ### Logging ###
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 400

    ### Hyperparameters ###
    # basic
    EPOCH_NUM: int = 50000
    LR: float = 0.001
    BATCH_SIZE: int = 256
    TEST_ALPHA: float = 0.4

    # optimizer
    WEIGHT_DECAY: float = 0.
    STEP_LR_STEP_SIZE: int = 100
    STEP_LR_GAMMA: float = 0.98
    MAX_GRAD_NORM: float = math.inf

    # model
    N_LAYERS: int = 4

    # transformer
    D_MODEL: int = 16
    N_HEAD: int = 1
    DIM_FEEDFORWARD: int = 32
    MAX_SEQ_LENGTH: int = 8
    DROPOUT: float = 0.

    # mlp
    HIDDEN_SIZE: int = 256


def remove_comments(json_like):
    json_like = re.sub(r'//.*', '', json_like)
    json_like = re.sub(r'/\*.*?\*/', '', json_like, flags=re.DOTALL)
    return json_like


def load_params(path=None):
    if path is None:
        return
    with open(path) as f:
        json_like = remove_comments(f.read())
        params = json.loads(json_like)

    for key, value in params.items():
        setattr(Param, key.upper(), value)
    print("Loaded params from:", path)
