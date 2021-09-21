import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

SENTENCE_STAT = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'


class Vocab:
    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for special_token in [SENTENCE_STAT, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN]:
            self._word_to_id[special_token] = self._count
            self._id_to_word[self._count] = special_token
            self._count += 1

        for token in tokens:
            self._word_to_id[token] = self._count
            self._id_to_word[self._count] = token
            self._count += 1

    def word2id(self, word):
        return self._word_to_id.get(word, UNKNOWN_TOKEN)

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def translate(self, id_list):
        return "".join(self.id2word(int(word_id)) for word_id in id_list)


class miDataset(data.Dataset):
    def __init__(self, src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask):
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.tgt_mask = tgt_mask

    def __getitem__(self, index):
        return self.src_tensor[index], self.tgt_tensor[index], \
               self.src_lens[index], self.tgt_lens[index], self.tgt_mask[index]

    def __len__(self):
        return self.src_tensor[0].size(0)


class Config:
    emb_dim = 128
    hidden_size = 256
    vocab_size = 30
    batch_size = 2
    rand_unif_init_mag = 0.02
    trunc_norm_init_std = 1e-4
    eps = 1e-12

def init_weights(m):
    if isinstance(m, nn.Embedding):
        m.weight.data.normal_(std=Config.trunc_norm_init_std)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-Config.rand_unif_init_mag, Config.rand_unif_init_mag)
            elif 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data.fill_(0.)
                param.data[start: end].fill_(1.)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(std=config.trunc_norm_init_std)
        if m.bias is not None:
            m.bias.data.normal_(std=config.trunc_norm_init_std)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, batch_first=True, bidirectional=True)

        self.apply(init_weights)

        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)

    def forward(self, input_tensor, seq_lens):
        embedded = self.embed(input_tensor)  # B x L x emb_dim

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # B x L x hidden_size*2
        encoder_outputs = encoder_outputs.contiguous()

        encoder_features = encoder_outputs.view(-1, 2 * config.hidden_size)  # (B x L) x hidden_size*2
        encoder_features = self.linear(encoder_features)

        return encoder_outputs, encoder_features, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.reduce_c = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.apply(init_weights)

    def forward(self, hidden):
        hn, cn = hidden  # D*1 x B x hidden_size
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

    def forward(self, y_t_1, s_t_1):
        # B x emb_dim
        y_t_1_embedded = self.embed(y_t_1)

        # output: B x 1 x hidden_size
        # hidden[0]: 1 x B x hidden_size
        output, hidden = self.lstm(y_t_1_embedded.unsqueeze(1), s_t_1)

        final_dist = F.softmax(self.fc_out(output.squeeze(1)), dim=1)

        return final_dist, hidden


def data_process(src: list, tgt: list, vocab: Vocab):
    src_lens = [len(s) for s in src]
    tgt_lens = [len(t) + 2 for t in tgt]  # +2 for START & END token

    src_max_len = max(src_lens)
    tgt_max_len = max(tgt_lens)

    src_tensor = torch.zeros((len(src), src_max_len), dtype=torch.int64)
    tgt_tensor = torch.zeros((len(tgt), tgt_max_len), dtype=torch.int64)
    tgt_mask = torch.zeros((len(tgt), tgt_max_len), dtype=torch.int64)

    for b, s in enumerate(src):
        id_list = []
        for token in s:
            id_list.append(vocab.word2id(token))
        src_tensor[b][:len(id_list)] = torch.LongTensor(id_list)
        src_tensor[b][len(id_list):].fill_(vocab.word2id(PAD_TOKEN))

    for b, t in enumerate(tgt):
        id_list = [vocab.word2id(SENTENCE_STAT)]
        for token in t:
            id_list.append(vocab.word2id(token))
        id_list.append(vocab.word2id(SENTENCE_END))
        tgt_tensor[b][:len(id_list)] = torch.LongTensor(id_list)
        tgt_tensor[b][len(id_list):].fill_(vocab.word2id(PAD_TOKEN))
        tgt_mask[b][:len(id_list)].fill_(1.)


    return src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask


if __name__ == '__main__':
    tokens = 'abcdefghijklmnopqrstuvwxyz'
    vocab = Vocab()
    config = Config()
    encoder = Encoder()
    decoder = Decoder()
    reducer = ReduceState()
    source = ['name', 'sex', 'address', 'job', 'interest']
    target = ['dwx', 'male', 'hunan', 'student', 'learning']

    src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask = data_process(source, target, vocab)

    train_dataset = miDataset(src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask)

    data_loader = data.DataLoader(train_dataset, batch_size=config.batch_size)

    input_batch, target_batch, input_lens, target_lens, target_mask = next(iter(data_loader))

    # encoder_outputs: B x L_src x 2*hidden_size
    # encoder_features: B*L_src x 2*hidden_size
    # hidden[0]: D*1 x B x hidden_size
    encoder_outputs, encoder_features, hidden = encoder(input_batch, input_lens)

    # 1 x B x hidden_size
    s_t_1 = reducer(hidden)

    step_losses = []

    for i in range(0, target_lens.max()-1):
        y_t_1 = target_batch[:, i]

        final_dist, s_t_1 = decoder(y_t_1, s_t_1)

        # B x 1
        gold_probs = torch.gather(final_dist, 1, target_batch[:, i+1].unsqueeze(1))
        step_loss = -torch.log(gold_probs + config.eps)

        step_loss *= target_mask[:, i+1]

