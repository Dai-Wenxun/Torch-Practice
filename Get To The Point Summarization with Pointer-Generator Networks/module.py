import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import config


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.W_s_b_attn = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)

        self.v = nn.Linear(config.hidden_size * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_features, encoder_pad_mask):
        src_len = encoder_outputs.size(1)

        # B x L x hidden_size*2
        decoder_features = self.W_s_b_attn(s_t_hat).unsqueeze(1).expand(-1, src_len, config.hidden_size*2).contiguous()
        # (B x L) x hidden_size*2
        decoder_features = decoder_features.view(-1, config.hidden_size * 2)
        # (B x L) x hidden_size*2
        attn_features = encoder_features + decoder_features
        # B x L
        e = self.v(F.tanh(attn_features)).view(-1, src_len)
        attn_dist = F.softmax(e, dim=1)*encoder_pad_mask

        norms_factor = attn_dist.sum(1, keepdim=True)
        attn_dist /= norms_factor

        # B x hidden_size*2
        c_t = torch.bmm(attn_dist.unsqueeze(1), encoder_outputs).view(-1, config.hidden_size * 2)

        return c_t


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, batch_first=True, bidirectional=True)

        self.apply(init_weights)

        self.W_h = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)

    def forward(self, src_tensor, src_lens):
        embedded = self.embed(src_tensor)  # B x L x emb_dim

        packed = pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # B x L x hidden_size*2
        encoder_outputs = encoder_outputs.contiguous()

        encoder_features = encoder_outputs.view(-1, config.hidden_size * 2)  # (B x L) x hidden_size*2
        encoder_features = self.W_h(encoder_features)

        return encoder_outputs, encoder_features, hidden  # hidden[0]: 2 x B x hidden_size


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.reduce_c = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.apply(init_weights)

    def forward(self, hidden):
        hn, cn = hidden  # 2 x B x hidden_size
        hn = hn.transpose(0, 1).contiguous().view(-1, config.hidden_size * 2)
        cn = cn.transpose(0, 1).contiguous().view(-1, config.hidden_size * 2)

        # B x hidden_size
        h_reduced = F.relu(self.reduce_h(hn))
        c_reduced = F.relu(self.reduce_c(cn))

        # 1 x B x hidden_size
        return h_reduced.unsqueeze(0), c_reduced.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, batch_first=True)
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(init_weights)

        self.context = nn.Linear(config.hidden_size * 2 + config.emb_dim, config.emb_dim)
        self.attention = Attention()

    def forward(self, y_t_1, s_t_1, c_t_1, encoder_outputs, encoder_features, encoder_pad_mask):
        # B x emb_dim
        y_t_1_embedded = self.embed(y_t_1)
        x = self.context(torch.cat((y_t_1_embedded, c_t_1), 1))

        # output: B x 1 x hidden_size
        # s_t[0]: 1 x B x hidden_size
        output, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        s_t_hat = torch.cat((s_t[0].view(-1, config.hidden_size), s_t[1].view(-1, config.hidden_size)), 1)

        # B x hidden_size*2
        c_t = self.attention(s_t_hat, encoder_outputs, encoder_features, encoder_pad_mask)

        final_dist = F.softmax(self.fc_out(output.squeeze(1)), dim=1)

        return final_dist, s_t, c_t  # final_dist: B x vocab_size


def init_weights(m):
    if isinstance(m, nn.Embedding):
        m.weight.data.normal_(std=config.trunc_norm_init_std)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data.fill_(0.)
                param.data[start: end].fill_(1.)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(std=config.trunc_norm_init_std)
        if m.bias is not None:
            m.bias.data.normal_(std=config.trunc_norm_init_std)
