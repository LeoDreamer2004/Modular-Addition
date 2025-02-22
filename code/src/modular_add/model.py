from typing import Optional

import torch.nn as nn
from torch import Tensor

from modular_add.params import *


def get_model(vocal_size: int) -> nn.Module:
    print("Using model type:", Param.MODEL)
    match Param.MODEL:
        case "transformer":
            return TransformerModel(
                vocal_size, d_model=Param.D_MODEL, n_head=Param.N_HEAD, n_layers=Param.N_LAYERS,
                max_seq_length=Param.MAX_SEQ_LENGTH, dim_feedforward=Param.DIM_FEEDFORWARD, dropout=Param.DROPOUT
            ).to(DEVICE)
        case "mlp":
            return MLPModel(vocal_size, hidden_size=Param.HIDDEN_SIZE).to(DEVICE)
        case "lstm":
            return LSTMModel(
                vocal_size, n_layers=Param.N_LAYERS, hidden_size=Param.HIDDEN_SIZE, dropout=Param.DROPOUT
            ).to(DEVICE)
    raise ValueError("Invalid model type")


class MLPModel(nn.Module):
    """
    MLP model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The number of features in the hidden state.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super(MLPModel, self).__init__()
        self.model_type = "MLP"
        self.token_embedding = nn.Linear(4 * vocab_size, hidden_size)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
        )
        self.layer_norm = nn.LayerNorm(hidden_size // 2)
        self.fc = nn.Linear(hidden_size // 2, vocab_size, bias=False)

    def forward(self, src: Tensor) -> Tensor:
        x = self.token_embedding(src.reshape(src.shape[0], -1))
        x = self.hidden(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    """
    LSTM model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        n_layers (int): The number of LSTM layers.
        input_size (int): The number of expected features of the input.
        hidden_size (int): The number of features in the hidden state.
        dropout (float): The dropout rate.
    """

    def __init__(self, vocab_size: int, n_layers: int, input_size: int = 128, hidden_size: int = 256,
                 dropout: float = 0):
        super(LSTMModel, self).__init__()
        self.model_type = "LSTM"
        self.token_embedding = nn.Linear(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src: Tensor) -> Tensor:
        if len(src.size()) == 2:
            seq_len = src.size(0)
            dim_len = 0
        else:
            seq_len = src.size(1)
            dim_len = 1

        x = self.token_embedding(src)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        x = x.sum(dim_len) / seq_len
        return x


class DecoderLayer(nn.Module):
    """
    Decoder layer for Transformer model.

    Args:
        d_model (int): The number of expected features of the input.
        n_head (int): The number of attention heads.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float = 0):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask=None) -> Tensor:
        # self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # feedforward
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The number of expected features of the input.
        n_head (int): The number of attention heads.
        dim_feedforward (int): The dimension of the feedforward network model.
        n_layers (int): The number of Transformer decoder layers.
        max_seq_length (int): The maximum sequence length.
        dropout (float, optional): The dropout rate. Default is 0.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_head: int,
            dim_feedforward: int,
            n_layers: int,
            max_seq_length: int,
            dropout: float = 0,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.token_embedding = nn.Linear(vocab_size, d_model)  # Embedding layer

        # Positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pos_embedding', pe)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, dim_feedforward, dropout) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.mask: Optional[Tensor] = None

    @staticmethod
    def _generate_square_subsequent_mask(size: int):
        mask = (torch.triu(torch.ones(size, size, device=DEVICE)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src: Tensor) -> Tensor:
        if len(src.size()) == 2:
            seq_len = src.size(0)
            dim_len = 0
        else:
            seq_len = src.size(1)
            dim_len = 1
        embed: Tensor = self.token_embedding(src)
        embed: Tensor = embed + self.pos_embedding[:, :seq_len, :]
        embed.transpose_(0, 1)

        if self.mask is None or self.mask.size(0) != seq_len:
            mask = self._generate_square_subsequent_mask(seq_len)
            self.mask = mask

        x = embed
        for layer in self.decoder_layers:
            x = layer(x, self.mask)

        x.transpose_(0, 1)
        if Param.LAYER_NORM:
            x = self.layer_norm(x)
        x = self.fc(x)
        x = x.sum(dim_len) / seq_len
        return x
