from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from modular_add.params import DEVICE

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
            temp = [0] * len(self)
            temp[self.s2i[s]] = 1
            result.append(temp)
        return torch.tensor(result, dtype=torch.float32)

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

    input: "x_1 + x_2 + ... + x_k ="            encoded shape: (4, vocab_size)  (one-hot encoding)
    output: (x_1 + x_2 + ... + x_k) % modulus   encoded shape: ()  (index)
    """

    def __init__(self, modulus: int, num_adder: int = 2):
        self.modulus = modulus
        self.num_adder = num_adder
        self.tokenizer = AlgorithmDataTokenizer(modulus)
        lhs, rhs = self.make_data()
        self.lhs = [self.tokenizer.encode(d).to(DEVICE) for d in lhs]
        self.rhs = torch.tensor([self.tokenizer.s2i[d] for d in rhs]).to(DEVICE)

    def __len__(self):
        return len(self.lhs)

    def __getitem__(self, idx):
        return self.lhs[idx], self.rhs[idx]

    @staticmethod
    def data_maker(n: int, modulus: int) -> List[Tuple]:
        if n <= 0:
            raise ValueError("number of adders should be more than 0.")
        if n == 1:
            return [(i,) for i in range(modulus)]
        results = []
        temp = AlgorithmDataSet.data_maker(n - 1, modulus)
        for i in range(modulus):
            for item in temp:
                results.append((*item, i))
        return results

    def make_data(self):
        all_nums = self.data_maker(self.num_adder, self.modulus)
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
