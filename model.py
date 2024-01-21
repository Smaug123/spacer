from typing import Tuple, Iterable

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SpaceInsertionModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length):
        super(SpaceInsertionModel, self).__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(2, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          batch_first=True)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.positional_encoding(self.src_embedding(src))  # + self.positional_encoding[:, :src.size(1), :]
        tgt = self.positional_encoding(self.tgt_embedding(tgt))  # + self.positional_encoding[:, :tgt.size(1), :]
        mask_size = tgt.size(1)
        output = self.transformer(src, tgt,
                                  tgt_mask=nn.Transformer.generate_square_subsequent_mask(mask_size).to('mps'))
        return self.output_linear(output).squeeze(-1)


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add more information here if needed
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


