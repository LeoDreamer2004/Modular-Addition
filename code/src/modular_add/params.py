import torch
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Param:
    ### Environment ###
    MODEL: str = "transformer"
    MODEL_PATH: str = "../model/transformer.pth"
    LOAD_MODEL: bool = False
    FIGURE_SAVE_PATH: str = "../fig/"
    RESULT_SAVE_PATH: str = "../result/"
    MODULUS: int = 97
    NUM_ADDER: int = 2
    SEED: int = 0
    PRELOAD_TO_DEVICE: bool = True
    FIG_SUFFIX: str = None
    USE_TF32: bool = False  # On supported hardware, use tf32 is faster and less power-consuming, but the result may be different.

    ### Logging ###
    LOG_INTERVAL: int = 10
    SAVE_MODEL_INTERVAL: int = 400
    SAVE_FIG_INTERVAL: int = 400

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
    MAX_GRAD_VALUE: float = float("inf")
    MOMENTUM: float = 0.9
    T_MAX: int = 1000,
    MIN_LR: float = 0.
    RMSPROP_ALPHA: float = 0.99
    RMSPROP_MOMENTUM: float = 0.9

    # scheduler
    SCHEDULER = "constant"
    STEP_LR_STEP_SIZE: int = 100
    STEP_LR_GAMMA: float = 0.98
    LAMBDA_LR_FUNC: str = ""

    # transformer
    D_MODEL: int = 16
    N_HEAD: int = 1
    DIM_FEEDFORWARD: int = 32
    MAX_SEQ_LENGTH: int = 2 * NUM_ADDER + 2
    DROPOUT: float = 0.
    LAYER_NORM: float = False

    # mlp
    HIDDEN_SIZE: int = 256


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
