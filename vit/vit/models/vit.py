import torch.nn as nn
import torch
from createPatches import CreatePatches
from attentionBlock import AttentionBlock


class ViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            in_channels: int = 3,
            patch_size: int = 16,
            embed_dim: int = 768,
            hidden_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            dropout: float = 0.0,
            num_classes: int = 1000
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Positional encoding. the + 1 is used as classification (cls) token
        # normally distributed
        # the first '1' means that for every batch, we have to treat the items together one by one
        # length of ( num_patches + 1 ) --> number of items generated
        # dimension is 'embed_dim'
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.attn_layers = nn.ModuleList(
            [AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        # 'apply' applies a given function to each module of a neural net
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # truncated normal distribution (re-sample outliers to enable smooth start of training)
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patches(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x[:, 0]

        return self.head(x)
