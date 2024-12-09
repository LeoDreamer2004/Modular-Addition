# We focus on x + y (mod p), tokenize?

import torch
from torch import Tensor
from torch.utils.data import Dataset

MODULUS = 13
NUMS = list(range(20))
EOS_TOKEN = "<eos>"
OP_TOKENS = ["+", "="]


class AlgorithmDataTokenizer:

    def __init__(self):
        self.i2s = self.gen_tokens()
        self.s2i = {s: i for i, s in enumerate(self.i2s)}

    def __len__(self):
        """
        Returns the number of tokens in the vocabulary
        """
        return len(self.i2s)

    def encode(self, eq: str):
        """Encode the equation into a tensor

        Arguments:
            eq -- equation to encode
        Returns:
            tensor representation of the equation
        """
        return torch.tensor([self.s2i[s] for s in eq.split()], dtype=torch.int)

    def decode(self, eq: Tensor):
        """Decode the tensor into a string

        Arguments:
            eq -- tensor to decode
        Returns:
            string representation of the tensor
        """
        return " ".join([self.i2s[s] for s in eq])

    @staticmethod
    def gen_tokens():
        return list(map(str, NUMS)) + OP_TOKENS + [EOS_TOKEN]


class AlgorithmDataSet(Dataset):

    def __init__(self):
        self.tokenizer = AlgorithmDataTokenizer()
        self.data = [self.tokenizer.encode(d) for d in self.make_data()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_data():
        return [
            f"{EOS_TOKEN} {a} + {b} = {(a + b) % MODULUS} {EOS_TOKEN}"
            for a in NUMS
            for b in NUMS
        ]


if __name__ == "__main__":
    tokenizer = AlgorithmDataTokenizer()
    eq = "1 + 2 = 3"
    encode = tokenizer.encode(eq)
    print(encode)
    decode = tokenizer.decode(encode)
    assert eq == decode

    from torch.utils.data import DataLoader
    dataloader = DataLoader(AlgorithmDataSet(), batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch)
        break