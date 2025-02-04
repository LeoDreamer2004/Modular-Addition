import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader

from modular_add.data import AlgorithmDataSet, NoneRandomDataloader
from modular_add.model import get_model
from modular_add.optim import get_optimizer, get_scheduler
from modular_add.params import *
from modular_add.util import compress_size, save_data, accuracy


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
    model = get_model(len(dataset.tokenizer)).train()
    if Param.COMPILE:
        try:
            model = torch.compile(model, dynamic=False)
        except RuntimeError:
            print("Compile failed. Using the original model.")
        else:
            print("Model compiled.")
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracy_list = []
    test_losses = []
    test_accuracy_list = []
    trained_epoch = 0
    compressed_size_list = []

    if Param.LOAD_MODEL:
        load_model(model)
        print("Model loaded from", Param.MODEL_PATH)

    # Start training
    try:
        if Param.CALCULATE_COMPLEXITY:
            compressed_size = compress_size(model, train_dataloader_val, Param.COMPLEXITY_TOL)
            compressed_size_list.append(compressed_size)
            print(f"Initial compressed size: {compressed_size}")

        for epoch in range(Param.EPOCH_NUM):
            epoch_loss = 0
            for lhs, rhs in train_dataloader:
                if not Param.PRELOAD_TO_DEVICE:
                    lhs = lhs.to(DEVICE)
                    rhs = rhs.to(DEVICE)

                optimizer.zero_grad()
                output: Tensor = model.forward(lhs)  # Type: ignore

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
                save_data(trained_epoch, train_losses, train_accuracy_list, test_losses, test_accuracy_list)

            trained_epoch += 1

            if Param.CALCULATE_COMPLEXITY and (epoch + 1) % Param.COMPLEXITY_INTERVAL == 0:
                compressed_size = compress_size(model, train_dataloader_val, Param.COMPLEXITY_TOL)
                compressed_size_list.append(compressed_size)
                print(f"Compressed size: {compressed_size}")

    except KeyboardInterrupt:
        print("Training interrupted.")
    else:
        print("Training finished.")

    save_model(model)
    print("Model saved at", Param.MODEL_PATH)

    # Save data
    save_data(trained_epoch, train_losses, train_accuracy_list, test_losses, test_accuracy_list)
    compressed_size_list = np.array(compressed_size_list)
    np.save(os.path.join(Param.RESULT_SAVE_PATH, "compressed_size.npy"), compressed_size_list)
