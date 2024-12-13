from params import *
from data import AlgorithmDataSet
from model import TransformerModel

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def load_model():
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


def save_model(model: TransformerModel):
    torch.save(model.state_dict(), MODEL_PATH)


def train():
    dataset = AlgorithmDataSet()
    # dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    train_data, test_data = train_test_split(dataset, test_size=TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=True)
    
    n_token = len(dataset.tokenizer)

    # FIXME: n_input = ?
    model = TransformerModel(n_token, 8, 1, 1, 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(train_dataloader):
            x: Tensor = data[0].to(DEVICE)
            y: Tensor = data[1].to(DEVICE)

            optimizer.zero_grad()
            # FIXME: the output shape is not correct
            output = model.forward(x)
            loss = criterion.forward(output, y)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")


train()
