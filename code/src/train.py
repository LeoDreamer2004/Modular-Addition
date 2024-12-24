import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modular_add.data import AlgorithmDataSet
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


def accuracy(label: Tensor, target: Tensor, model: nn.Module):
    with torch.no_grad():
        output = model.forward(label)
        _, predicted = torch.max(output, 1)
        loss = F.cross_entropy(output, target)
        correct = (predicted == target).sum().item()  # Type: ignore
        return loss.item(), correct / target.size(0)


def train():
    setup()

    # Initialize data
    dataset = AlgorithmDataSet(Param.MODULUS, Param.NUM_ADDER)
    print("Modulus:", Param.MODULUS)
    print("Dataset initialized. Data size:", len(dataset))
    train_data, test_data = train_test_split(dataset, test_size=Param.TEST_ALPHA)
    print("Train size:", len(train_data), "Test size:", len(test_data))
    train_dataloader = DataLoader(train_data, batch_size=Param.BATCH_SIZE, shuffle=True)

    # FIXME: If not preload to device, rewrite the code, see `Param.PRELOAD_TO_DEVICE`
    # Don't using Dataloader to avoid random action when calculating accuracy
    # Otherwise, the result may not be reproducible with different log interval
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

    if Param.LOAD_MODEL:
        load_model(model)
        print("Model loaded from", Param.MODEL_PATH)

    # Start training
    try:
        for epoch in range(Param.EPOCH_NUM):
            epoch_loss = 0
            for lhs, rhs in train_dataloader:
                if not Param.PRELOAD_TO_DEVICE:
                    print("Not preload to device is not supported yet.")
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
    if Param.DROPOUT > 0:
        suffix = f"{Param.OPTIMIZER.lower()}-{Param.TEST_ALPHA}-{Param.NUM_ADDER}-dropout"
    else:
        suffix = f"{Param.OPTIMIZER.lower()}-{Param.TEST_ALPHA}-{Param.NUM_ADDER}"
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
