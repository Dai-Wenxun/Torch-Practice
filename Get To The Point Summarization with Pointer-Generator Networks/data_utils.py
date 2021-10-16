import os
import collections
import pickle

from enum_type import SpecialTokens

def load_data(dataset_path, max_length):
    with open(dataset_path, "r", encoding='utf-8') as fin:
        text = []
        for line in fin:
            line = line.strip().lower()
            words = line.split()
            text.append(words[:max_length])
    return text


def build_vocab(text, max_vocab_size, special_token_list):
    word_list = list()
    for group in text:
        for doc in group:
            for word in doc:
                if word not in ['nostalgia', 'gone', 'water', 'wine', 'cherish']:
                    word_list.append(word)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))
    return idx2token, token2idx, max_vocab_size


def text2idx(source_text, target_text, token2idx, pointer_gen=False):
    data_dict = {'source_idx': [], 'source_length': [],
                 'input_target_idx': [], 'output_target_idx': [], 'target_length': []}

    if pointer_gen:
        data_dict['extended_source_idx'] = []
        data_dict['oovs_list'] = []

    sos_idx = token2idx[SpecialTokens.SOS]
    eos_idx = token2idx[SpecialTokens.EOS]
    unknown_idx = token2idx[SpecialTokens.UNK]

    for source_sent, target_sent in zip(source_text, target_text):
        source_idx = [token2idx.get(word, unknown_idx) for word in source_sent]
        input_target_idx = [sos_idx] + [token2idx.get(word, unknown_idx) for word in target_sent]

        if pointer_gen:
            extended_source_idx, oovs = article2ids(source_sent, token2idx, unknown_idx)
            output_target_idx = abstract2ids(target_sent, oovs, token2idx, unknown_idx) + [eos_idx]
        else:
            output_target_idx = [token2idx.get(word, unknown_idx) for word in target_sent] + [eos_idx]

        data_dict['source_idx'].append(source_idx)
        data_dict['source_length'].append(len(source_idx))
        data_dict['input_target_idx'].append(input_target_idx)
        data_dict['output_target_idx'].append(output_target_idx)
        data_dict['target_length'].append(len(input_target_idx))

        if pointer_gen:
            data_dict['extended_source_idx'].append(extended_source_idx)
            data_dict['oovs_list'].append(oovs)

    return data_dict


def article2ids(article_words, token2idx, unknown_idx):
    ids = []
    oovs = []
    for w in article_words:
        i = token2idx.get(w, unknown_idx)
        if i == unknown_idx:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(token2idx) + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, article_oovs, token2idx, unknown_idx):
    ids = []
    for w in abstract_words:
        i = token2idx.get(w, unknown_idx)
        if i == unknown_idx:
            if w in article_oovs:
                vocab_idx = len(token2idx) + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unknown_idx)
        else:
            ids.append(i)
    return ids


def _pad_batch_sequence(self, text_idx_data, idx_length_data):
    max_len = max(idx_length_data)
    new_data = []
    for seq, len_seq in zip(text_idx_data, idx_length_data):
        new_data.append(seq + [self.padding_token_idx] * (max_len - len_seq))
    new_data = torch.LongTensor(new_data)
    length = torch.LongTensor(idx_length_data)
    return new_data, length

def _get_extra_zeros(self, oovs_list):
    max_oovs_num = max([len(oovs) for oovs in oovs_list])
    extra_zeros = torch.zeros(len(oovs_list), max_oovs_num)
    return extra_zeros