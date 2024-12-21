from torch.utils.data import DataLoader

from modular_add.data import AlgorithmDataSet


def get_nums(lhs: str):
    return int(lhs.split()[0]), int(lhs.split()[2])


def test_dataset():
    dataset = AlgorithmDataSet(13)
    for i in range(len(dataset)):
        lhs, rhs = dataset[i]
        lhs = dataset.tokenizer.decode(lhs)
        # rhs = dataset.tokenizer.decode(rhs)
        num1, num2 = get_nums(lhs)
        assert (num1 + num2) % 13 == int(rhs)


def test_dataloader():
    dataset = AlgorithmDataSet(13)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    for i, (lhs, rhs) in enumerate(dataloader):
        batch_size = lhs.shape[0]  # Type: ignore
        for j in range(batch_size):
            lhs_j = dataset.tokenizer.decode(lhs[j])  # Type: ignore
            num1, num2 = get_nums(lhs_j)
            assert (num1 + num2) % 13 == int(rhs[j])
