from sklearn.model_selection import train_test_split
from torch import nn, optim, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import random
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
            return MLPModel(n_token, n_layers=Param.N_LAYERS, hidden_size=Param.HIDDEN_SIZE).to(DEVICE)


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    print("Using optimizer:", Param.OPTIMIZER)
    match Param.OPTIMIZER.lower():
        case "adam":
            return optim.Adam(model.parameters(), lr=Param.LR)
        case "adamw":
            return optim.AdamW(model.parameters(), lr=Param.LR, weight_decay=Param.WEIGHT_DECAY)
        case "sgd":
            return optim.SGD(model.parameters(), lr=Param.LR, momentum=0.9)


def save_model(model: nn.Module):
    if not os.path.exists(os.path.dirname(Param.MODEL_PATH)):
        os.makedirs(os.path.dirname(Param.MODEL_PATH))
    torch.save(model.state_dict(), Param.MODEL_PATH)


def accuracy(label: Tensor, target: Tensor, model: nn.Module):
    with torch.no_grad():
        output = model.forward(label)
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()  # Type: ignore
        return correct / target.size(0)


def train():
    seed()

    # Initialize data
    dataset = AlgorithmDataSet(Param.MODULUS)
    print("Modulus:", Param.MODULUS)
    print("Dataset initialized. Data size:", len(dataset))
    train_data, test_data = train_test_split(dataset, test_size=Param.TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=Param.BATCH_SIZE, shuffle=True)
    train_label = torch.stack([lhs for lhs, _ in train_data]).to(DEVICE)
    train_target = torch.stack([rhs for _, rhs in train_data]).to(DEVICE)
    test_label = torch.stack([lhs for lhs, _ in test_data]).to(DEVICE)
    test_target = torch.stack([rhs for _, rhs in test_data]).to(DEVICE)

    # Prepare model
    model = get_model(len(dataset.tokenizer))
    optimizer = get_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1 / (1 + e) ** 0.1)
    losses = []
    train_accuracy_list = []
    test_accuracy_list = []
    trained_epoch = 0

    # Start training
    try:
        for epoch in range(Param.EPOCH_NUM):
            epoch_loss = 0
            for lhs, rhs in train_dataloader:
                optimizer.zero_grad()
                output = model.forward(lhs)  # Type: ignore

                loss = criterion.forward(output, rhs)
                epoch_loss += loss.item()
                loss.backward()
                clip_grad_norm_(model.parameters(), Param.MAX_GRAD_NORM)
                optimizer.step()
            scheduler.step()
            losses.append(epoch_loss)

            if (epoch + 1) % Param.LOG_INTERVAL == 0:
                train_accuracy = accuracy(train_label, train_target, model)
                test_accuracy = accuracy(test_label, test_target, model)
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

            trained_epoch += 1
    except KeyboardInterrupt:
        print("Training interrupted.")
    else:
        print("Training finished.")

    save_model(model)
    print("Model saved at", Param.MODEL_PATH)

    # Save figures
    if not os.path.exists(Param.FIGURE_SAVE_PATH):
        os.makedirs(Param.FIGURE_SAVE_PATH)
    suffix = f"{Param.MODEL}-{Param.OPTIMIZER}-{Param.TEST_ALPHA}"
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.savefig(os.path.join(Param.FIGURE_SAVE_PATH, f"loss-{suffix}.png"))
    plt.clf()
    x = range(Param.LOG_INTERVAL, trained_epoch + 1, Param.LOG_INTERVAL)
    if len(x) > len(train_accuracy_list):
        x = x[:len(train_accuracy_list)]
    plt.plot(x, train_accuracy_list, label="train")
    plt.plot(x, test_accuracy_list, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig(os.path.join(Param.FIGURE_SAVE_PATH, f"accuracy-{suffix}.png"))
    print("Figures saved at", Param.FIGURE_SAVE_PATH)
