import torch
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Param:
    ### Environment ###
    MODEL: str = "transformer"
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

    # model
    N_LAYERS: int = 4

    # optimizer
    OPTIMIZER: str = "adam"
    WEIGHT_DECAY: float = 0.
    MAX_GRAD_NORM: float = float("inf")
    MOMENTUM: float = 0.9
    RMSPROP_ALPHA: float = 0.99
    RMSPROP_MOMENTUM: float = 0.9

    # scheduler
    SCHEDULER = "constant"
    STEP_LR_STEP_SIZE: int = 100
    STEP_LR_GAMMA: float = 0.98
    LAMBDA_LR_FUNC: str = "decay_transformer"

    # transformer
    D_MODEL: int = 16
    N_HEAD: int = 1
    DIM_FEEDFORWARD: int = 32
    MAX_SEQ_LENGTH: int = 8
    DROPOUT: float = 0.

    # mlp
    HIDDEN_SIZE: int = 256

    # draw
    DRAW_CLIP: int = 0


def load_params(path=None):
    import json
    import re

    def remove_comments(json_like):
        json_like = re.sub(r'//.*', '', json_like)
        json_like = re.sub(r'/\*.*?\*/', '', json_like, flags=re.DOTALL)
        return json_like

    if path is None:
        return
    with open(path) as f:
        json_like = remove_comments(f.read())
        params = json.loads(json_like)

    for key, value in params.items():
        setattr(Param, key.upper(), value)
    print("Loaded params from:", path)
