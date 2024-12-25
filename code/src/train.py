import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader

from modular_add.data import AlgorithmDataSet, NoneRandomDataloader
from modular_add.model import get_model
from modular_add.optim import get_optimizer, get_scheduler
from modular_add.params import *


def setup():
    torch.manual_seed(Param.SEED)
    torch.cuda.manual_seed(Param.SEED)
    np.random.seed(Param.SEED)
    if Param.USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    random.seed(Param.SEED)


def save_model(model: nn.Module):
    if not os.path.exists(os.path.dirname(Param.MODEL_PATH)):
        os.makedirs(os.path.dirname(Param.MODEL_PATH))
    torch.save(model.state_dict(), Param.MODEL_PATH)


def load_model(model: nn.Module):
    model.load_state_dict(torch.load(Param.MODEL_PATH, weights_only=True))


def accuracy(dataloader: NoneRandomDataloader, model: nn.Module):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        loss = 0.0
        for lhs, rhs in dataloader:
            if not Param.PRELOAD_TO_DEVICE:
                lhs = lhs.to(DEVICE)
                rhs = rhs.to(DEVICE)
            output = model.forward(lhs)
            loss += F.cross_entropy(output, rhs).item()
            _, predicted = output.max(1)
            total += rhs.size(0)
            correct += predicted.eq(rhs).sum().item()
        model.train()
    return loss, correct / total


def save_fig(trained_epoch: int, train_losses: List, train_acc: List, test_losses: List, test_acc: List):
    save_path = os.path.join(Param.FIGURE_SAVE_PATH, Param.MODEL.lower())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if Param.FIG_SUFFIX is not None:
        suffix = f"{Param.OPTIMIZER.lower()}-{Param.TEST_ALPHA}-{Param.NUM_ADDER}-{Param.FIG_SUFFIX}"
    else:
        suffix = f"{Param.OPTIMIZER.lower()}-{Param.TEST_ALPHA}-{Param.NUM_ADDER}"
    x = range(Param.LOG_INTERVAL, trained_epoch + 1 + Param.LOG_INTERVAL, Param.LOG_INTERVAL)
    x = x[:len(train_acc)]
    plt.plot(x, train_losses, label="train")
    plt.plot(x, test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"loss-{suffix}.png"), dpi=300)
    plt.clf()

    plt.plot(x, train_acc, label="train")
    plt.plot(x, test_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig(os.path.join(save_path, f"accuracy-{suffix}.png"), dpi=300)
    plt.clf()


def train():
    setup()

    # Initialize data
    dataset = AlgorithmDataSet(Param.MODULUS, Param.NUM_ADDER)
    print("Modulus:", Param.MODULUS)
    print("Dataset initialized. Data size:", len(dataset))
    train_data, test_data = train_test_split(dataset, test_size=Param.TEST_ALPHA)
    print("Train size:", len(train_data), "Test size:", len(test_data))
    train_dataloader = DataLoader(train_data, batch_size=Param.BATCH_SIZE, shuffle=True)

    # Don't use Dataloader to avoid random action when calculating accuracy
    # Otherwise, the result may not be reproducible with different log interval
    batch_size = len(train_data) if Param.PRELOAD_TO_DEVICE else Param.BATCH_SIZE
    train_dataloader_val = NoneRandomDataloader(train_data, batch_size=batch_size)
    test_dataloader_val = NoneRandomDataloader(test_data, batch_size=len(test_data))

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

    if Param.LOAD_MODEL:
        load_model(model)
        print("Model loaded from", Param.MODEL_PATH)

    # Start training
    try:
        for epoch in range(Param.EPOCH_NUM):
            epoch_loss = 0
            for lhs, rhs in train_dataloader:
                if not Param.PRELOAD_TO_DEVICE:
                    lhs = lhs.to(DEVICE)
                    rhs = rhs.to(DEVICE)

                optimizer.zero_grad()
                output = model.forward(lhs)  # Type: ignore

                loss = criterion.forward(output, rhs)
                epoch_loss += loss.item()
                loss.backward()
                clip_grad_norm_(model.parameters(), Param.MAX_GRAD_NORM)
                clip_grad_value_(model.parameters(), Param.MAX_GRAD_VALUE)
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % Param.LOG_INTERVAL == 0:
                train_loss, train_accuracy = accuracy(train_dataloader_val, model)
                test_loss, test_accuracy = accuracy(test_dataloader_val, model)
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

            if (epoch + 1) % Param.SAVE_MODEL_INTERVAL == 0:
                save_model(model)
                print("Saved model at epoch", epoch + 1)

            if (epoch + 1) % Param.SAVE_FIG_INTERVAL == 0:
                save_fig(trained_epoch, train_losses, train_accuracy_list, test_losses, test_accuracy_list)

            trained_epoch += 1
    except KeyboardInterrupt:
        print("Training interrupted.")
    else:
        print("Training finished.")

    save_model(model)
    print("Model saved at", Param.MODEL_PATH)

    # Save figures
    save_fig(trained_epoch, train_losses, train_accuracy_list, test_losses, test_accuracy_list)
