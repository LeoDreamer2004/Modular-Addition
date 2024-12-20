from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from modular_add.data import AlgorithmDataSet
from modular_add.model import TransformerModel
from modular_add.params import *


def load_model(n_token: int) -> TransformerModel:
    model = TransformerModel(n_token, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
                             max_seq_length=MAX_SEQ_LENGTH, dim_feedforward=DIM_FEEDFORWARD).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


def save_model(model: TransformerModel):
    torch.save(model.state_dict(), MODEL_PATH)


def accuracy(data_loader: DataLoader, model: nn.Module) -> float:
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
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = AlgorithmDataSet(MODULUS)
    train_data, test_data = train_test_split(dataset, test_size=TEST_ALPHA)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    n_token = len(dataset.tokenizer)

    model = TransformerModel(n_token, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
                             max_seq_length=MAX_SEQ_LENGTH, dim_feedforward=DIM_FEEDFORWARD).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

    losses = []
    train_accuracy_list = []
    test_accuracy_list = []
    log_interval = 10

    for epoch in range(EPOCH_NUM):
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

        if (epoch + 1) % log_interval == 0:
            train_accuracy = accuracy(train_dataloader, model)
            test_accuracy = accuracy(test_dataloader, model)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            print(
                f"Epoch: {epoch + 1}, Loss: {epoch_loss:.6e}, Train accuracy: {train_accuracy * 100:.4f} %, Test accuracy: {test_accuracy * 100:.4f} %")

    save_model(model)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.show()
    x = range(1, EPOCH_NUM + 1, log_interval)
    plt.plot(x, train_accuracy_list, label="train")
    plt.plot(x, test_accuracy_list, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    train()
