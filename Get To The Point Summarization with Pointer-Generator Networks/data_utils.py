import os
import collections
import pickle


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
            word_list.extend(doc)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))
    return idx2token, token2idx, max_vocab_size


def dump_data(data_path, source_text_data, target_text_data, idx2token, token2idx, source_suffix="", target_suffix=""):
        vocab_file = os.path.join(data_path, 'vocab.bin')
        with open(vocab_file, "wb") as f_vocab:
            pickle.dump([idx2token, token2idx], f_vocab)

        for i, prefix in enumerate(['train', 'dev', 'test']):
            source_text = source_text_data[i]
            target_text = target_text_data[i]
            source_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, source_suffix))
            target_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, target_suffix))
            with open(source_file, "wb") as f_text:
                pickle.dump(source_text, f_text)
            with open(target_file, "wb") as f_text:
                pickle.dump(target_text, f_text)


def load_restored(data_path, source_suffix="", target_suffix=""):
    source_text_data = []
    target_text_data = []
    for prefix in ['train', 'dev', 'test']:
        source_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, source_suffix))
        target_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, target_suffix))
        with open(source_file, "rb") as f_text:
            text = pickle.load(f_text)
            source_text_data.append(text)
        with open(target_file, "rb") as f_text:
            text = pickle.load(f_text)
            target_text_data.append(text)

    vocab_file = os.path.join(data_path, 'vocab.bin')
    with open(vocab_file, "rb") as f_vocab:
        idx2token, token2idx = pickle.load(f_vocab)

    return source_text_data, target_text_data, idx2token, token2idx

def _article2ids(self, article_words):
    ids = []
    oovs = []
    unk_id = self.unknown_token_idx
    for w in article_words:
        i = self.token2idx.get(w, unk_id)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(self.vocab_size + oov_num)
        else:
            ids.append(i)
    return ids, oovs

def _abstract2ids(self, abstract_words, article_oovs):
    ids = []
    unk_id = self.unknown_token_idx
    for w in abstract_words:
        i = self.token2idx.get(w, unk_id)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = self.vocab_size + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids