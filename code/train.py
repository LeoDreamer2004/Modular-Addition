from params import *
from data import AlgorithmDataSet

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def train():
    dataset = AlgorithmDataSet()
    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    train_data, test_data = train_test_split(dataset, test_size=TEST_ALPHA)


train()
