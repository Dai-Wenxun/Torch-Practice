import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LuongAttention(nn.Module):
    def __init__(self, source_size, target_size, alignment_method='concat'):
        super(LuongAttention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.alignment_method = alignment_method

        if self.alignment_method == 'general':
            self.energy_linear = nn.Linear(target_size, source_size, bias=False)
        elif self.alignment_method == 'concat':
            self.energy_linear = nn.Linear(source_size + target_size, target_size)
            self.v = nn.Parameter(torch.rand(target_size, dtype=torch.float32))
        elif self.alignment_method == 'dot':
            assert self.source_size == target_size
        else:
            raise ValueError("The alignment method for Luong Attention must be in ['general', 'concat', 'dot'].")

    def score(self, decoder_hidden_states, encoder_outputs):
        tgt_len = decoder_hidden_states.size(1)
        src_len = encoder_outputs.size(1)

        if self.alignment_method == 'general':
            energy = self.energy_linear(decoder_hidden_states)
            encoder_outputs = encoder_outputs.permute(0, 2, 1)
            energy = energy.bmm(encoder_outputs)
            return energy
        elif self.alignment_method == 'concat':
            decoder_hidden_states = decoder_hidden_states.unsqueeze(2).repeat(1, 1, src_len, 1)  # B * tgt_len * src_len * target_size
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, tgt_len, 1, 1)
            energy = torch.tanh(self.energy_linear(torch.cat((decoder_hidden_states, encoder_outputs), dim=-1)))
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

    def forward(self, decoder_hidden_states, encoder_outputs, encoder_masks):
        tgt_len = decoder_hidden_states.size(1)
        energy = self.score(decoder_hidden_states, encoder_outputs)
        probs = F.softmax(energy, dim=-1) * encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)
        return context, probs


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_enc_layers, dropout_ratio, bidirectional=True):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_enc_layers = num_enc_layers
        self.bidirectional = bidirectional

        self.encoder = nn.LSTM(embedding_size, hidden_size, num_enc_layers,
                               batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, source_embeddings, source_length):
        packed_source_embeddings = pack_padded_sequence(source_embeddings, source_length.cpu(),
                                                        batch_first=True, enforce_sorted=False)

        encoder_outputs, encoder_hidden_states = self.encoder(packed_source_embeddings)

        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, encoder_hidden_states










class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()





    def forward(self, ):
