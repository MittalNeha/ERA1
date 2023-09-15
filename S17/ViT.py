import torch
import torch.nn as nn
import math

from transformer import Encoder, EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock, PatchEmbedding


class ViT(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: PatchEmbedding,
                 src_pos: nn.Parameter, class_embed, projection_layer: nn.Sequential) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.class_embedding = class_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        batch_size = src.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        src = self.src_embed(src)  # patch_embedding
        src = torch.cat((class_token, src), dim=1)
        src = src + self.src_pos  # PositionalEmbedding
        # Dropout?
        return self.encoder(src, src_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, x, mask=None):
        x = self.encode(x, mask)
        x = self.project(x[:, 0])
        return x


def build_ViT(img_size: int = 224, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768,
              N: int = 12, n_heads: int = 12, dropout: float = 0.1, mlp_size: int = 3072, attn_dropout: float = 0,
              num_classes: int = 1000) -> ViT:
    assert img_size % patch_size == 0, f"Image size needs to be divisible by patch_size| img_size: {img_size}, patch_size: {patch_size}"
    num_patches = (img_size * img_size) // patch_size ** 2

    class_embedding = nn.Parameter(data=torch.randn((1, 1, embedding_dim), requires_grad=True))
    # src_embed = InputEmbeddings(embedding_dim, src_vocab_size)
    src_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size,
                               embedding_dim=embedding_dim)  # patch_embedding
    # src_pos = PositionalEncoding(embedding_dim, src_seq_len, dropout)
    src_pos = nn.Parameter(
        data=torch.randn((1, num_patches + 1, embedding_dim), requires_grad=True))  # position_embedding

    encoder_blocks = []
    for _ in range(int(N)):
        encoder_self_attention_block = MultiHeadAttentionBlock(embedding_dim, n_heads, dropout=attn_dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, d_ff=mlp_size, dropout=0.1, activation='gelu')
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks), norm_first=False)

    classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                               nn.Linear(in_features=embedding_dim,
                                         out_features=num_classes)
                               )

    transformer = ViT(encoder, src_embed, src_pos, class_embedding, classifier)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
