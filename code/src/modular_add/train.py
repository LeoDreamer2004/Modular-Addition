from modular_add.params import *
from modular_add.data import AlgorithmDataSet
from modular_add.model import TransformerModel

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
    dataset = AlgorithmDataSet(MODULUS)
    # dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    train_data, test_data = train_test_split(dataset, test_size=TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    n_token = len(dataset.tokenizer)

    model = TransformerModel(n_token, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
                             max_seq_length=MAX_SEQ_LENGTH, dim_feedforward=DIM_FEEDFORWARD).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH_NUM):
        for i, (lhs, rhs) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model.forward(lhs)  # Type: ignore
            loss = criterion.forward(output[:, -1, :].reshape(-1), rhs.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")


if __name__ == "__main__":
    train()
