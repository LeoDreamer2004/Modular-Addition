import bz2
import gc
import os
import pickle
from copy import deepcopy
from typing import List

import torch
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import functional as F

from modular_add.data import NoneRandomDataloader
from modular_add.params import Param, DEVICE


@torch.no_grad()
def coarse_grain(param: Tensor, delta: float, data_ratio: float) -> Tensor:
    if param.dim() == 1:
        return torch.round(param / delta) * delta
    U, S, Vt = torch.linalg.svd(param)
    rank = S.size(0)
    total = S.sum()
    current = 0
    for i in range(S.size(0)):
        current += S[i]
        if current / total >= data_ratio:
            rank = i + 1
            break
    low_rank_param = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
    return torch.round(low_rank_param / delta) * delta


def compress_size(model: nn.Module, train_dataloader: NoneRandomDataloader, tol: float) -> int:
    """
    Compress the model using the Bayesian optimization algorithm.

    Args:
        model: Model to compress.
        train_dataloader: Dataloader for training data.
        tol: Tolerance for the distortion.

    Returns:
        The minimum size of the compressed model.
    """
    original_loss = 0

    with torch.no_grad():
        for lhs, rhs in train_dataloader:
            output = model.forward(lhs)
            original_loss += F.cross_entropy(output, rhs).item()

    test_model = deepcopy(model)
    test_model.eval()
    compressed_size = 0

    for original_param, test_param in zip(model.parameters(), test_model.parameters()):
        def objective(delta: float, data_ratio: float = 1.0):
            with torch.no_grad():
                coarse_params = coarse_grain(original_param.data, delta, data_ratio)
                test_param.data = coarse_params

            loss = 0
            with torch.no_grad():
                for l, r in train_dataloader:
                    out = test_model.forward(l)
                    loss += F.cross_entropy(out, r).item()

            distortion = abs(loss - original_loss)
            if distortion > tol:
                return -1e38
            size = -len(bz2.compress(pickle.dumps(test_param.data)))
            return size

        if original_param.dim() == 1:
            bounds = {"delta": (0.0001, 0.02)}
        else:
            bounds = {"delta": (0.0001, 0.02), "data_ratio": (0.9, 1.0)}

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=1,
            verbose=0
        )
        optimizer.maximize(init_points=2, n_iter=30)

        min_size = -optimizer.max["target"]
        if min_size < 1e37:
            compressed_size += min_size
        else:
            # If the distortion is too large, we do not compress the parameter.
            compressed_size += len(bz2.compress(pickle.dumps(original_param.data)))
        test_param.data = original_param.data

        # Garbage collection
        del optimizer
        gc.collect()

    torch.cuda.empty_cache()
    del test_model
    return compressed_size


def save_data(trained_epoch: int, train_losses: List, train_acc: List, test_losses: List, test_acc: List):
    save_path = os.path.join(Param.FIGURE_SAVE_PATH, Param.MODEL.lower())
    result_path = os.path.join(Param.RESULT_SAVE_PATH, Param.MODEL.lower())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

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

    with open(os.path.join(result_path, f"accuracy-{suffix}.txt"), "w") as f:
        for i in range(len(train_acc)):
            f.write(f"{x[i]} {train_acc[i]} {test_acc[i]}\n")


@torch.no_grad()
def accuracy(dataloader: NoneRandomDataloader, model: nn.Module):
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
