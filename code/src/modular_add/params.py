import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model/transformer.pth"

# Hyperparameters
EPOCH_NUM = 50000
LEARNING_RATE = 0.001
BATCH_SIZE = 200
TEST_ALPHA = 0.5
D_MODEL = 16
N_HEAD = 1
DIM_FEEDFORWARD = 32
N_LAYERS = 4
MAX_SEQ_LENGTH = 8
DROPOUT = 0.
MODULUS = 97
