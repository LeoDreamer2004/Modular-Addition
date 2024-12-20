import torch
import json
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Param:
    MODEL_PATH = "../model/transformer.pth"
    MODULUS = 47

    # Default Hyperparameters
    SEED = 0
    EPOCH_NUM = 50000
    LR = 0.001
    BATCH_SIZE = 256
    TEST_ALPHA = 0.4
    D_MODEL = 16
    N_HEAD = 1
    DIM_FEEDFORWARD = 32
    N_LAYERS = 4
    MAX_SEQ_LENGTH = 8
    DROPOUT = 0.
    STEP_LR_STEP_SIZE = 100
    STEP_LR_GAMMA = 0.98
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 100


def load_params(path=None):

    if path is None:
        return 
    with open(path) as f:
        params = json.load(f)
    print("Loaded params from:", path)
    
    Param.MODEL_PATH = params.get("model_path", Param.MODEL_PATH)
    Param.MODULUS = params.get("modulus", Param.MODULUS)
    Param.EPOCH_NUM = params.get("epoch_num", Param.EPOCH_NUM)
    Param.LR = params.get("lr", Param.LR)
    Param.BATCH_SIZE = params.get("batch_size", Param.BATCH_SIZE)
    Param.TEST_ALPHA = params.get("test_alpha", Param.TEST_ALPHA)
    Param.D_MODEL = params.get("d_model", Param.D_MODEL)
    Param.N_HEAD = params.get("n_head", Param.N_HEAD)
    Param.DIM_FEEDFORWARD = params.get("dim_feedforward",Param.DIM_FEEDFORWARD)
    Param.N_LAYERS = params.get("n_layers", Param.N_LAYERS)
    Param.MAX_SEQ_LENGTH = params.get("max_seq_length", Param.MAX_SEQ_LENGTH)
    Param.DROPOUT = params.get("dropout", Param.DROPOUT)
    Param.STEP_LR_STEP_SIZE = params.get("step_lr_step_size", Param.STEP_LR_STEP_SIZE)
    Param.STEP_LR_GAMMA = params.get("step_lr_gamma", Param.STEP_LR_GAMMA)
    Param.LOG_INTERVAL = params.get("log_interval", Param.LOG_INTERVAL)
    Param.SAVE_INTERVAL = params.get("save_interval", Param.SAVE_INTERVAL)