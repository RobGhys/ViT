import torch.nn as nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim: int = 768, hidden_dim: int = 3072,
                 num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.pre_norm(x)
        # MultiheadAttention returns attention output and weights,
        # we need only the outputs, so [0] index.
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        x = x + self.MLP(self.norm(x))

        return x


if __name__ == '__main__':
    embed_dim: int = 768
    hidden_dim: int = 3072
    num_heads: int = 12
    dropout: float = 0.0
    batch_size = 1

    model = AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
    input_data = torch.rand(batch_size, num_heads, embed_dim)
    output = model(input_data)

    print(f'output shape: {output.shape}')

    print('Test Passed')
