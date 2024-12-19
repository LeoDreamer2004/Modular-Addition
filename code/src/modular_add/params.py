import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model/transformer.pth"

# Hyperparameters
EPOCH_NUM = 10000
LEARNING_RATE = 0.005
BATCH_SIZE = 50
TEST_ALPHA = 0.2
D_MODEL = 16
N_HEAD = 8
DIM_FEEDFORWARD = 32
N_LAYERS = 4
MAX_SEQ_LENGTH = 8
DROPOUT = 0.
MODULUS = 13
