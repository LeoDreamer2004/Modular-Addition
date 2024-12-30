import bz2
import os
import pickle
from copy import deepcopy
from typing import List

import optuna
import torch
from matplotlib import pyplot as plt
from optuna import Trial
from torch import nn
from torch.nn import functional as F

from modular_add.data import NoneRandomDataloader
from modular_add.params import Param


def compress_size(model: nn.Module, train_dataloader_val: NoneRandomDataloader, epsilon: float):
    def coarse_grain(params, delta, rank):
        quantized_params = torch.round(params / delta) * delta
        if quantized_params.dim() == 1:
            return quantized_params
        U, S, Vt = torch.linalg.svd(quantized_params)
        rank = min(rank, params.size(0), params.size(1))
        return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    original_loss = 0

    with torch.no_grad():
        for lhs, rhs in train_dataloader_val:
            output = model.forward(lhs)
            original_loss += F.cross_entropy(output, rhs).item()

    def objective(trial: Trial):
        delta = trial.suggest_float("delta", 0.0001, 0.01)
        rank = trial.suggest_int("rank", 100, 512)
        model_cloned = deepcopy(model)

        with torch.no_grad():
            for p in model_cloned.parameters():
                coarse_params = coarse_grain(p, delta, rank)
                p.data = coarse_params

        loss = 0
        with torch.no_grad():
            for l, r in train_dataloader_val:
                out = model_cloned.forward(l)
                loss += F.cross_entropy(out, r).item()

        distortion = abs(loss - original_loss)
        if distortion > epsilon:
            return 1e38
        size = len(bz2.compress(pickle.dumps(model_cloned.state_dict())))
        del model_cloned
        return size

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200, n_jobs=10)
    compressed_size = study.best_value
    if compressed_size >= 1e37:
        compressed_size = len(bz2.compress(pickle.dumps(model.state_dict())))
    # Garbage collection
    del study
    torch.cuda.empty_cache()

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
