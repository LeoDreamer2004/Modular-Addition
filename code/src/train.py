from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import random
import argparse
import os

from modular_add.data import AlgorithmDataSet
from modular_add.model import MLPModel, TransformerModel
from modular_add.params import *


def seed():
    torch.manual_seed(Param.SEED)
    np.random.seed(Param.SEED)
    random.seed(Param.SEED)


def get_model(n_token: int) -> nn.Module:
    print("Using model type:", Param.MODEL)
    match Param.MODEL:
        case "transformer":
            return TransformerModel(
                n_token, d_model=Param.D_MODEL, n_head=Param.N_HEAD, n_layers=Param.N_LAYERS,
                max_seq_length=Param.MAX_SEQ_LENGTH, dim_feedforward=Param.DIM_FEEDFORWARD
            ).to(DEVICE)
        case "mlp":
            return MLPModel(n_token, Param.N_LAYERS).to(DEVICE)


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    match Param.OPTIM:
        case "adam":
            return optim.Adam(model.parameters(), lr=Param.LR)
        case "sgd":
            return optim.SGD(model.parameters(), lr=Param.LR, momentum=0.9)


def save_model(model: nn.Module):
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
    print("Modulus:", Param.MODULUS)
    print("Dataset initialized. Data size: ", len(dataset))
    train_data, test_data = train_test_split(dataset, test_size=Param.TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=Param.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)  # full-batch

    # Prepare model
    model = get_model(len(dataset.tokenizer))
    optimizer = get_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Param.STEP_LR_STEP_SIZE, gamma=Param.STEP_LR_GAMMA
    )

    losses = []
    train_accuracy_list = []
    test_accuracy_list = []

    # Start training
    for epoch in range(Param.EPOCH_NUM):
        epoch_loss = 0
        for lhs, rhs in train_dataloader:
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

    # Plot the result
    save_model(model)
    print("Training finished.")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.savefig(os.path.join(Param.FIGURE_SAVE_PATH, "loss.png"))
    plt.clf()
    x = range(1, Param.EPOCH_NUM + 1, Param.LOG_INTERVAL)
    plt.plot(x, train_accuracy_list, label="train")
    plt.plot(x, test_accuracy_list, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig("acc.png")
