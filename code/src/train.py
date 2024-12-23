import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modular_add.data import AlgorithmDataSet
from modular_add.model import get_model
from modular_add.optim import get_optimizer, get_scheduler
from modular_add.params import *


def seed():
    torch.manual_seed(Param.SEED)
    torch.cuda.manual_seed(Param.SEED)
    np.random.seed(Param.SEED)
    random.seed(Param.SEED)


def save_model(model: nn.Module):
    if not os.path.exists(os.path.dirname(Param.MODEL_PATH)):
        os.makedirs(os.path.dirname(Param.MODEL_PATH))
    torch.save(model.state_dict(), Param.MODEL_PATH)


def accuracy(label: Tensor, target: Tensor, model: nn.Module):
    with torch.no_grad():
        output = model.forward(label)
        _, predicted = torch.max(output, 1)
        loss = F.cross_entropy(output, target)
        correct = (predicted == target).sum().item()  # Type: ignore
        return loss.item(), correct / target.size(0)


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
    scheduler = get_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracy_list = []
    test_losses = []
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

            if (epoch + 1) % Param.LOG_INTERVAL == 0:
                train_loss, train_accuracy = accuracy(train_label, train_target, model)
                test_loss, test_accuracy = accuracy(test_label, test_target, model)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
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
    save_path = os.path.join(Param.FIGURE_SAVE_PATH, Param.MODEL.lower())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    suffix = f"{Param.OPTIMIZER.lower()}-{Param.TEST_ALPHA}"
    x = range(Param.LOG_INTERVAL, trained_epoch + 1, Param.LOG_INTERVAL)
    x = x[:len(train_accuracy_list)]
    plt.plot(x, train_losses, label="train")
    plt.plot(x, test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"loss-{suffix}.png"), dpi=300)
    plt.clf()

    plt.plot(x, train_accuracy_list, label="train")
    plt.plot(x, test_accuracy_list, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig(os.path.join(save_path, f"accuracy-{suffix}.png"), dpi=300)
    print("Figures saved at", save_path)
