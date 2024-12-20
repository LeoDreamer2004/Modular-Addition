from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import random
import argparse

from modular_add.data import AlgorithmDataSet
from modular_add.model import TransformerModel
from modular_add.params import *


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help="The path to the params")
    namespace = parser.parse_args()
    path = namespace.param
    load_params(path)


def seed():
    torch.manual_seed(Param.SEED)
    np.random.seed(Param.SEED)
    random.seed(Param.SEED)


def load_model(n_token: int):
    model = TransformerModel(n_token, d_model=Param.D_MODEL, n_head=Param.N_HEAD, n_layers=Param.N_LAYERS,
                             max_seq_length=Param.MAX_SEQ_LENGTH, dim_feedforward=Param.DIM_FEEDFORWARD).to(DEVICE)
    model.load_state_dict(torch.load(Param.MODEL_PATH))
    return model


def save_model(model: TransformerModel):
    torch.save(model.state_dict(), Param.MODEL_PATH)


def accuracy(data_loader: DataLoader, model: nn.Module):
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (lhs, rhs) in enumerate(data_loader):
            labels = rhs.argmax(dim=2).reshape(-1)
            output = model.forward(lhs)
            _, predicted = torch.max(output[:, -1, :], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Type: ignore
        return correct / total


def train():
    seed()

    # Initialize data
    dataset = AlgorithmDataSet(Param.MODULUS)
    print("Dataset initialized. Data size: ", len(dataset))
    train_data, test_data = train_test_split(dataset, test_size=Param.TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=Param.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    n_token = len(dataset.tokenizer)
    model = TransformerModel(n_token, d_model=Param.D_MODEL, n_head=Param.N_HEAD, n_layers=Param.N_LAYERS,
                             max_seq_length=Param.MAX_SEQ_LENGTH, dim_feedforward=Param.DIM_FEEDFORWARD).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Param.LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Param.STEP_LR_STEP_SIZE, gamma=Param.STEP_LR_GAMMA)

    losses = []
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(Param.EPOCH_NUM):
        epoch_loss = 0
        for i, (lhs, rhs) in enumerate(train_dataloader):
            labels = rhs.argmax(dim=2).reshape(-1)
            optimizer.zero_grad()
            output = model.forward(lhs)  # Type: ignore    

            loss = criterion.forward(output[:, -1, :], labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        losses.append(epoch_loss)

        if (epoch + 1) % Param.LOG_INTERVAL == 0:
            train_accuracy = accuracy(train_dataloader, model)
            test_accuracy = accuracy(test_dataloader, model)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            print(
                f"Epoch: {epoch + 1}",
                f"Loss: {epoch_loss:.6e}", 
                f"Train accuracy: {train_accuracy * 100:.4f}%",
                f"Test accuracy: {test_accuracy * 100:.4f}%"
            )
            
        if (epoch + 1) % Param.SAVE_INTERVAL == 0:
            save_model(model)
            print("Saved model at epoch", epoch + 1)

    save_model(model)
    print("Training finished.")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.savefig("loss.png")
    x = range(1, Param.EPOCH_NUM + 1, Param.LOG_INTERVAL)
    plt.plot(x, train_accuracy_list, label="train")
    plt.plot(x, test_accuracy_list, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig("acc.png")


if __name__ == "__main__":
    init_args()
    train()
