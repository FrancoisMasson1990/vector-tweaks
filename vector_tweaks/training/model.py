import torch.nn.functional as F

from torch import nn


class EmbeddingAlign(nn.Module):
    def __init__(self, emb_dim: int = 1536) -> None:
        super(EmbeddingAlign, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        transformed = self.linear(x)
        normalized = F.normalize(transformed, p=2, dim=-1)
        return normalized
