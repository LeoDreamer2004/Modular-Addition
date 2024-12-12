import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
from typing import Optional


class PositionalEncoding(nn.Module):

    def __init__(self, n_input: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, n_input)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, n_input, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / n_input)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Args:
        n_token (int): The size of the vocabulary.
        n_input (int): The number of input features.
        n_head (int): The number of attention heads.
        n_hidden (int): The number of hidden units in the feedforward layers.
        n_layers (int): The number of Transformer encoder layers.
        dropout (float, optional): The dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        n_token: int,
        n_input: int,
        n_head: int,
        n_hidden: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.src_mask: Optional[Tensor] = None
        self.pos_encoder = PositionalEncoding(n_input, dropout)
        encoder_layers = TransformerEncoderLayer(n_input, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, n_input)
        self.n_input = n_input
        self.decoder = nn.Linear(n_input, n_token)

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * torch.sqrt(torch.tensor(self.n_input))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
