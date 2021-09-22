import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


SENTENCE_STAT = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'


class Vocab:
    def __init__(self, tokens):
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

    def vocab_size(self):
        return self._count

    def STAT_id(self):
        return self.word2id(SENTENCE_STAT)

    def END_id(self):
        return self.word2id(SENTENCE_END)

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
        return self.src_tensor.size(0)


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