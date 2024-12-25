from modular_add.data import AlgorithmDataSet, NoneRandomDataloader
from modular_add.params import DEVICE


def get_nums(lhs: str):
    return int(lhs.split()[0]), int(lhs.split()[2])


def test_none_random_dataloader():
    dataset = AlgorithmDataSet(11, 2)
    data_list = [dataset[i] for i in range(len(dataset))]
    dataloader = NoneRandomDataloader(data_list, batch_size=64)
    for lhs, rhs in dataloader:
        assert lhs.device.type == DEVICE.type
        assert rhs.device.type == DEVICE.type
        batch_size = lhs.shape[0]
        for j in range(batch_size):
            lhs_j = dataset.tokenizer.decode(lhs[j])
            num1, num2 = get_nums(lhs_j)
            assert (num1 + num2) % 11 == int(rhs[j])
        break
