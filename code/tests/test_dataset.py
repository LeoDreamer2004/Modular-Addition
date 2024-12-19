from modular_add.data import AlgorithmDataSet


def get_nums(lhs: str):
    return int(lhs.split()[0]), int(lhs.split()[2])


def test_dataset():
    dataset = AlgorithmDataSet(13)
    for i in range(len(dataset)):
        lhs, rhs = dataset[i]
        lhs = dataset.tokenizer.decode(lhs)
        rhs = dataset.tokenizer.decode(rhs)
        num1, num2 = get_nums(lhs)
        assert (num1 + num2) % 13 == int(rhs)
