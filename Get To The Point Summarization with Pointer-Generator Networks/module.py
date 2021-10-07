import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_enc_layers, dropout_ratio, bidirectional=True):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_enc_layers,
                               batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, source_embeddings, source_length):
        packed_source_embeddings = pack_padded_sequence(source_embeddings, source_length.cpu(),
                                                        batch_first=True, enforce_sorted=False)

        encoder_outputs, encoder_hidden_states = self.encoder(packed_source_embeddings)

        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, encoder_hidden_states


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_size,
            context_size,
            num_dec_layers,
            dropout_ratio=0.0,
            alignment_method='concat',
            is_coverage=True
    ):
        super(Decoder, self).__init__()
        self.is_coverage = is_coverage
        self.context_size = context_size

        self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers,
                               batch_first=True, dropout=dropout_ratio)

        self.attention = LuongAttention(context_size, hidden_size, alignment_method, is_coverage)
        self.x_context = nn.Linear(embedding_size + context_size, embedding_size)
        self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)
        self.vocab_linear = nn.Linear(hidden_size, vocab_size)
        self.p_gen_linear = nn.Linear(context_size + hidden_size + embedding_size, 1)

    def forward(self, input_embeddings, context, decoder_hidden_states, encoders_outputs, encoder_masks,
                extra_zeros, extended_source_idx, coverage=None):
        final_vocab_dists = []
        attn_dists = []
        coverages = []
        dec_length = input_embeddings.size(1)

        p_gen = None
        for step in range(dec_length):
            step_input_embeddings = input_embeddings[:, step, :].unsqueeze(1)  # B x 1 x 128

            x = self.x_context(torch.cat((step_input_embeddings, context), dim=-1))  # B x 1 x 128

            step_decoder_outputs, decoder_hidden_states = self.decoder(x, decoder_hidden_states)  # B x 1 x 256

            context, attn_dist, coverage = self.attention(step_decoder_outputs, encoders_outputs,
                                                          encoder_masks, coverage)  # B x 1 x src_len

            vocab_dist = self.vocab_linear(self.attention_dense(torch.cat((step_decoder_outputs, context), dim=-1)))
            vocab_dist = F.softmax(vocab_dist.squeeze(1), dim=-1)  # B x vocab_size

            p_gen_input = torch.cat((context, step_decoder_outputs, x), dim=-1)  # B x 1 x (256 + 256 + 128)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input).squeeze(2))  # B x 1

            attn_dist_ = (1 - p_gen) * attn_dist.squeeze(1)  # B x src_len

            extended_vocab_dist = torch.cat(((vocab_dist * p_gen), extra_zeros), dim=1)  # B x (vocab_size+max_oovs_num)

            final_vocab_dist = extended_vocab_dist.scatter_add(1, extended_source_idx, attn_dist_)

            final_vocab_dists.append(final_vocab_dist.unsqueeze(1))
            attn_dists.append(attn_dist)
            coverages.append(coverage)

        final_vocab_dists = torch.cat(final_vocab_dists, dim=1)  # B x dec_len x (vocab_size+max_oovs_num)
        attn_dists = torch.cat(attn_dists, dim=1)  # B x dec_len x src_len
        if self.is_coverage:
            coverages = torch.cat(coverages, dim=1)  # B x dec_len x src_len

        return final_vocab_dists, context, decoder_hidden_states, attn_dists, p_gen, coverages


class LuongAttention(nn.Module):
    def __init__(self, source_size, target_size, alignment_method='concat', is_coverage=True):
        super(LuongAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.alignment_method = alignment_method
        self.is_coverage = is_coverage

        if self.is_coverage:
            self.coverage_linear = nn.Linear(1, target_size, bias=False)

        if self.alignment_method == 'general':
            self.energy_linear = nn.Linear(target_size, source_size, bias=False)
        elif self.alignment_method == 'concat':
            self.energy_linear = nn.Linear(source_size + target_size, target_size)
            self.v = nn.Parameter(torch.rand(target_size, dtype=torch.float32))
        elif self.alignment_method == 'dot':
            assert self.source_size == target_size
        else:
            raise ValueError(
                "The alignment method for Luong Attention must be in ['general', 'concat', 'dot'].")

    def score(self, decoder_hidden_states, encoder_outputs, coverage):
        tgt_len = decoder_hidden_states.size(1)
        src_len = encoder_outputs.size(1)

        if self.is_coverage:
            coverage = self.coverage_linear(coverage.unsqueeze(3))

        if self.alignment_method == 'general':
            energy = self.energy_linear(decoder_hidden_states)
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = energy.bmm(encoder_outputs)
            return energy
        elif self.alignment_method == 'concat':
            # B * tgt_len * src_len * target_size
            decoder_hidden_states = decoder_hidden_states.unsqueeze(2).repeat(1, 1, src_len, 1)
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, tgt_len, 1, 1)
            energy = self.energy_linear(torch.cat((decoder_hidden_states, encoder_outputs), dim=-1))
            if self.is_coverage:
                energy = energy + coverage
            energy = torch.tanh(energy)
            energy = self.v.mul(energy).sum(dim=-1)
            return energy
        elif self.alignment_method == 'dot':
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = decoder_hidden_states.bmm(encoder_outputs)
            return energy
        else:
            raise NotImplementedError(
                "No such alignment method {} for computing Luong scores.".format(self.alignment_method)
            )

    def forward(self, decoder_hidden_states, encoder_outputs, encoder_masks, coverage=None):
        tgt_len = decoder_hidden_states.size(1)
        energy = self.score(decoder_hidden_states, encoder_outputs, coverage)
        probs = F.softmax(energy, dim=-1) * encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)

        if self.is_coverage:
            coverage = probs + coverage

        return context, probs, coverage
