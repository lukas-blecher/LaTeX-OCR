import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, emb_dim)
        nn.init.normal_(self.emb.weight, std = 0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device = x.device)
        return self.emb(n)[None, :, :]
