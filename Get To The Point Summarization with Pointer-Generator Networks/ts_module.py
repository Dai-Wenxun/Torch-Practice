import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, num_enc_layers, d_model, n_heads, d_ff, dropout_ratio):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_ratio)
                                     for _ in range(num_enc_layers)])

    def forward(self, enc_outputs, enc_self_attn_masks):
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_masks)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class TransformerDecoder(nn.Module):
    def __init__(self, num_dec_layers, d_model, n_heads, d_ff, dropout_ratio):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, d_ff, dropout_ratio)
                                     for _ in range(num_dec_layers)])

    def forward(self, dec_outputs, enc_outputs, dec_self_attn_masks, dec_enc_attn_masks):
        dec_self_attns = []
        dec_enc_attns = []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_masks, dec_enc_attn_masks)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.proj_query = nn.Linear(d_model, n_heads * self.d_k)
        self.proj_key = nn.Linear(d_model, n_heads * self.d_k)
        self.proj_value = nn.Linear(d_model, n_heads * self.d_v)
        self.proj_out = nn.Linear(n_heads * self.d_v, d_model)

    def score(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # B x n_heads x q_len x k_len

        scores = scores.masked_fill_(mask, -1e9)

        scores = F.softmax(scores, dim=-1)

        value = torch.matmul(scores, value)  # B x n_heads x q_len x d_v

        return value, scores

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query = self.proj_query(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.proj_key(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.proj_value(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        value, scores = self.score(query, key, value, mask)

        value = value.transpose(1, 2).view(batch_size, -1, self.n_heads * self.d_v)

        value = self.proj_out(value)

        return value, scores


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ratio):
        super(PoswiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs):
        return self.linear2(self.dropout(F.relu(self.linear1(inputs))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_ratio):
        super(TransformerEncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForward(d_model, d_ff, dropout_ratio)

        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, enc_outputs, enc_self_attn_masks):
        residual = enc_outputs
        enc_outputs, enc_self_attns = self.enc_self_attn(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_masks)
        enc_outputs = self.norm1(residual + self.dropout1(enc_outputs))

        residual = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.norm2(residual + self.dropout2(enc_outputs))

        return enc_outputs, enc_self_attns


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_ratio):
        super(TransformerDecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForward(d_model, d_ff, dropout_ratio)

        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_outputs, enc_outputs, dec_self_attn_masks, dec_enc_attn_masks):
        residual = dec_outputs
        dec_outputs, dec_self_attns = self.dec_self_attn(dec_outputs, dec_outputs, dec_outputs, dec_self_attn_masks)
        dec_outputs = self.norm1(residual + self.dropout1(dec_outputs))

        residual = dec_outputs
        dec_outputs, dec_enc_attns = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_masks)
        dec_outputs = self.norm2(residual + self.dropout2(dec_outputs))

        residual = dec_outputs
        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = self.norm3(residual + self.dropout3(dec_outputs))

        return dec_outputs, dec_self_attns, dec_enc_attns


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout_ratio, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])


