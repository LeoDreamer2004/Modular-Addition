from torch import nn, optim
from torch.optim import lr_scheduler

from modular_add.params import Param


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    print("Using optimizer:", Param.OPTIMIZER)
    match Param.OPTIMIZER.lower():
        case "adam":
            return optim.Adam(model.parameters(), lr=Param.LR)
        case "adamw":
            return optim.AdamW(model.parameters(), lr=Param.LR, weight_decay=Param.WEIGHT_DECAY)
        case "sgd":
            return optim.SGD(model.parameters(), lr=Param.LR, momentum=Param.MOMENTUM)


def get_scheduler(optimizer: optim.Optimizer) -> lr_scheduler.LRScheduler:
    print("Using scheduler:", Param.SCHEDULER)
    match Param.SCHEDULER.lower():
        case "constant":
            return lr_scheduler.ConstantLR(optimizer)
        case "step":
            return lr_scheduler.StepLR(
                optimizer, step_size=Param.STEP_LR_STEP_SIZE, gamma=Param.STEP_LR_GAMMA
            )
        case "lambda":
            return lr_scheduler.LambdaLR(optimizer, eval(Param.LAMBDA_LR_FUNC))
        case "constant":
            return lr_scheduler.ConstantLR(optimizer)


def decay_mlp(e):
    return 1 / (1 + e) ** 0.15


def decay_transformer(e):
    return 1 / (1 + e) ** 0.055


def decay_lstm(e):
    return 1 / (1 + e) ** 0.04
