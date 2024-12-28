from typing import Tuple, List

from matplotlib import pyplot as plt

RESULTS = {
    "../result/transformer/accuracy-adamw-0.6-2.txt": "40%",
    "../result/transformer/accuracy-adamw-0.55-2.txt": "45%",
    "../result/transformer/accuracy-adamw-0.5-2.txt": "50%",
    "../result/transformer/accuracy-adamw-0.4-2.txt": "60%",
}


def read_data(file_path: str) -> Tuple[List[float], List[float]]:
    x = []
    y = []
    with open(file_path, "r") as f:
        for line in f:
            epoch, train_acc, test_acc = line.strip().split(" ")
            x.append(float(train_acc))
            y.append(float(test_acc))
    return x, y


def plot_data(x: List[float], y: List[float], label: str, test_only=False) -> None:
    epochs = list(range(1, len(x) + 1))
    if test_only:
        plt.plot(epochs, y, label=f"{label}")
    else:
        lines = plt.plot(epochs, x, label=f"{label} Train", linewidth=0.8, linestyle="--")
        plt.plot(epochs, y, label=f"{label} Test", color=lines[0].get_color())


def main():
    for file_path, label in RESULTS.items():
        x, y = read_data(file_path)
        plot_data(x, y, label, test_only=False)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend()
    plt.savefig("../result/transformer/accuracy.png", dpi=300)


if __name__ == '__main__':
    main()
