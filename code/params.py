import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model/transformer.pth"
INPUT_SIZE = 7  # [<eos>, num, +, num, =, num, <eos>]

# Hyperparameters
EPOCH_NUM = 1000
BATCHSIZE = 100
TEST_ALPHA = 0.2
