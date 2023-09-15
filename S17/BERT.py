import torch
import torch.nn as nn
import math

from transformer import Encoder, EncoderBlock, PositionalEncoding, InputEmbeddings, ProjectionLayer, MultiHeadAttentionBlock, \
    FeedForwardBlock


class BERT(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)  # Embedding
        src = src + self.src_pos(src)  # PositionalEmbedding
        return self.encoder(src, src_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, x, mask=None):
        x = self.encode(x, mask)
        x = self.project(x)
        return x


def build_BERT(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, d_model: int = 512,
               N: int = 6, n_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> BERT:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_blocks = []
    for _ in range(int(N)):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = BERT(encoder, src_embed, src_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
