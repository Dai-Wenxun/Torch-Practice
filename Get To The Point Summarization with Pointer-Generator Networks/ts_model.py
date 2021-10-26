import torch
import torch.nn as nn

from model import AbstractModel
from ts_module import TransformerEncoder, TransformerDecoder, PositionalEmbedding


class Transformer(AbstractModel):
    def __init__(self, config):
        super(Transformer, self).__init__(config)

        self.d_model = config['d_model']
        self.d_ff = config['d_ff']
        self.n_heads = config['n_heads']
        self.dropout_ratio = config['dropout_ratio']

        self.positional_embedder = PositionalEmbedding(self.d_model, self.dropout_ratio)

        self.source_token_embedder = nn.Embedding(self.max_vocab_size, self.d_model,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.max_vocab_size, self.d_model,
                                                  padding_idx=self.padding_token_idx)

        self.encoder = TransformerEncoder(self.num_enc_layers, self.d_model, self.n_heads,
                                          self.d_ff, self.dropout_ratio)

        self.decoder = TransformerDecoder(self.num_dec_layers, self.d_model, self.n_heads,
                                          self.d_ff, self.dropout)



    def forward(self, corpus):
        # Encoder
        source_idx = corpus['source_idx']
        source_embeddings = self.positional_embedder(self.source_token_embedder(source_idx))
        enc_self_attn_masks = get_attn_pad_mask(source_idx, source_idx, self.padding_token_idx).to(self.device)
        enc_outputs, enc_self_attns = self.encoder(source_embeddings, enc_self_attn_masks)

        # Decoder
        input_target_idx = corpus['input_target_idx']
        input_embeddings = self.positional_embedder(self.target_token_embedder(input_target_idx))
        dec_self_attn_pad_masks = get_attn_pad_mask(input_target_idx, input_target_idx, self.padding_token_idx).to(self.device)
        dec_self_attn_subsequent_masks = get_attn_subsequent_mask(input_target_idx).to(self.device)
        dec_self_attn_masks = dec_self_attn_pad_masks + dec_self_attn_subsequent_masks
        dec_enc_attn_masks = get_attn_pad_mask(input_target_idx, source_idx, self.padding_token_idx).to(self.device)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(input_embeddings, enc_outputs, dec_self_attn_masks, dec_enc_attn_masks)





# helper functions
def get_attn_pad_mask(seq_q, seq_k, pad_id):
    mask = torch.eq(seq_k, pad_id).unsqueeze(1).repeat(1, seq_q.size(1), 1)
    return mask


def get_attn_subsequent_mask(seq):
    mask = torch.eq(torch.triu(torch.ones(seq.size(0), seq.size(1), seq.size(1)), 1), 1)
    return mask
