from itertools import product
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from modular_add.params import DEVICE, Param

EOS_TOKEN = "<eos>"
OP_TOKENS = ["+", "="]


class AlgorithmDataTokenizer:

    def __init__(self, modulus: int):
        self.modulus = modulus
        self.nums = list(range(modulus))
        self.operators = ["+", "="]
        self.i2s = self.gen_tokens()
        self.s2i = {s: i for i, s in enumerate(self.i2s)}

    def __len__(self):
        """
        Returns the number of tokens in the vocabulary
        """
        return len(self.i2s)

    def encode(self, equation: str) -> Tensor:
        """Encode the equation into a tensor using one-hot encoding

        Arguments:
            equation: equation to encode
        Returns:
            tensor representation of the equation
        """
        result = []
        for s in equation.split():
            one_hot = F.one_hot(torch.tensor(self.s2i[s]), num_classes=len(self))
            result.append(one_hot)
        return torch.stack(result).to(torch.float32)

    def decode(self, equation: Tensor) -> str:
        """Decode the tensor into a string

        Arguments:
            equation: tensor to decode
        Returns:
            string representation of the tensor
        """
        return " ".join([self.i2s[i] for i in equation.argmax(dim=1)])

    def gen_tokens(self):
        return list(map(str, self.nums)) + OP_TOKENS + [EOS_TOKEN]


class AlgorithmDataSet(Dataset):
    """
    Modular addition dataset

    input: "x_1 + x_2 + ... + x_k ="            encoded shape: (2 * k, vocab_size)  (one-hot encoding)
    output: (x_1 + x_2 + ... + x_k) % modulus   encoded shape: ()  (index)
    """

    def __init__(self, modulus: int, num_adder: int = 2):
        self.modulus = modulus
        self.num_adder = num_adder
        self.tokenizer = AlgorithmDataTokenizer(modulus)
        lhs, rhs = self.make_data()
        if Param.PRELOAD_TO_DEVICE:
            self.lhs = [self.tokenizer.encode(d).to(DEVICE) for d in lhs]
            self.rhs = torch.tensor([self.tokenizer.s2i[d] for d in rhs]).to(DEVICE)
        else:
            self.lhs = [self.tokenizer.encode(d) for d in lhs]
            self.rhs = torch.tensor([self.tokenizer.s2i[d] for d in rhs])

    def __len__(self):
        return len(self.lhs)

    def __getitem__(self, idx):
        return self.lhs[idx], self.rhs[idx]

    @staticmethod
    def data_maker(n: int, modulus: int) -> List[Tuple]:
        if n <= 0:
            raise ValueError("number of adders should be more than 0.")
        return list(product(range(modulus), repeat=n))

    def make_data(self):
        all_nums = AlgorithmDataSet.data_maker(self.num_adder, self.modulus)
        lhs = []
        rhs = []
        for nums in all_nums:
            rhs_sum = 0
            lhs_str = ""
            for num in nums:
                rhs_sum += num
                lhs_str += f"{num} + "
            lhs_str = lhs_str[:-2] + "="
            lhs.append(lhs_str)
            rhs.append(str(rhs_sum % self.modulus))
        return lhs, rhs


class NoneRandomDataloader:
    """
    Dataloader that does not shuffle the data and does not contain any randomness
    """

    def __init__(self, dataset: List[Tuple[Tensor]], batch_size: int = None):
        self.batch_size = len(dataset) if batch_size is None else batch_size
        self.dataset = dataset

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            end = min(i + self.batch_size, len(self.dataset))
            lhs = torch.stack([d[0] for d in self.dataset[i:end]])
            rhs = torch.stack([d[1] for d in self.dataset[i:end]])
            yield lhs, rhs
