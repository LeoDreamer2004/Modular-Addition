import argparse
from modular_add.params import load_params
from train import train


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", "-p", dest="param", type=str, help="The path to the params")
    namespace = parser.parse_args()
    path = namespace.param
    load_params(path)


if __name__ == "__main__":
    init_args()
    train()
