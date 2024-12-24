from torch.utils.data import DataLoader

from modular_add.data import AlgorithmDataSet
from modular_add.params import DEVICE


def get_nums(lhs: str):
    return int(lhs.split()[0]), int(lhs.split()[2])


def get_nums4(lhs: str):
    lhs = lhs.split()
    return int(lhs[0]), int(lhs[2]), int(lhs[4]), int(lhs[6])


def test_dataset():
    dataset = AlgorithmDataSet(13)
    for i in range(len(dataset)):
        lhs, rhs = dataset[i]
        assert (lhs[-1] == dataset.tokenizer.encode("=").to(DEVICE)).all()  # Type: ignore
        lhs = dataset.tokenizer.decode(lhs)
        assert lhs[-1] == "="
        num1, num2 = get_nums(lhs)
        assert (num1 + num2) % 13 == int(rhs)


def test_dataset_more_nums():
    modulus = 23
    dataset = AlgorithmDataSet(modulus, 4)
    assert len(dataset) == modulus ** 4
    for i in range(len(dataset)):
        lhs, rhs = dataset[i]
        lhs = dataset.tokenizer.decode(lhs)
        assert lhs[-1] == "="
        n1, n2, n3, n4 = get_nums4(lhs)
        assert (n1 + n2 + n3 + n4) % modulus == int(rhs)


def test_dataloader():
    dataset = AlgorithmDataSet(13)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    for i, (lhs, rhs) in enumerate(dataloader):
        batch_size = lhs.shape[0]  # Type: ignore
        for j in range(batch_size):
            lhs_j = dataset.tokenizer.decode(lhs[j])  # Type: ignore
            num1, num2 = get_nums(lhs_j)
            assert (num1 + num2) % 13 == int(rhs[j])
