import torch
from torch import nn, optim, Tensor
from torch.optim import lr_scheduler

from modular_add.params import Param


class SignSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, weight_decay: float = 0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step using SignSGD."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad_p: Tensor = p.grad

                if weight_decay != 0:
                    grad_p = grad_p.add(p, alpha=weight_decay)

                # Update using the sign of the gradient
                p.add_(grad_p.sign(), alpha=-lr)

        return loss


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    print("Using optimizer:", Param.OPTIMIZER)
    match Param.OPTIMIZER.lower():
        case "adam":
            return optim.Adam(model.parameters(), lr=Param.LR)
        case "adamw":
            return optim.AdamW(model.parameters(), lr=Param.LR, weight_decay=Param.WEIGHT_DECAY)
        case "sgd":
            return optim.SGD(model.parameters(), lr=Param.LR, momentum=Param.MOMENTUM)
        case "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=Param.LR,
                alpha=Param.RMSPROP_ALPHA,
                momentum=Param.RMSPROP_MOMENTUM,
                weight_decay=Param.WEIGHT_DECAY
            )
        case "signgd":
            return SignSGD(
                model.parameters(),
                lr=Param.LR,
                weight_decay=Param.WEIGHT_DECAY
            )
    raise ValueError("Invalid optimizer type")


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
            return lr_scheduler.ConstantLR(optimizer, factor=1)
        case "cosine":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=Param.T_MAX, eta_min=Param.MIN_LR)
    raise ValueError("Invalid scheduler type")


def transformer_sgd(e):
    if e < 600:
        return 1
    elif e < 4000:
        return 1.02 ** ((e - 600) // 10)
    return (1.02 ** 340) * 1.002 ** ((e - 4000) // 10)


def transformer_adam(e):
    if e < 1000:
        return 1
    return 0.99 ** ((e - 1000) // 10)


def transformer_adamw(e):
    ratio = 0.99 ** (e // 100)
    if ratio < 1e-2:
        return 1e-2
    return ratio
