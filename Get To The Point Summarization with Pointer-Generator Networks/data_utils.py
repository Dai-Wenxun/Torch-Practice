import os
import collections
import pickle


def load_data(dataset_path, max_seq_length):
    with open(dataset_path, "r", encoding='utf-8') as fin:
        text = []
        for line in fin:
            line = line.strip().lower()
            words = line.split()
            text.append(words[:max_seq_length])
    return text


def build_vocab(text_data_list, max_vocab_size, special_tokens):
    word_list = list()
    for text_data in text_data_list:
        for text in text_data:
            for words in text:
                word_list.append(words)
    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_tokens + tokens
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


def detect_restored(data_path, source_suffix="", target_suffix=""):
    required_files = []
    for prefix in ['train', 'dev', 'test']:
        source_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, source_suffix))
        target_file = os.path.join(data_path, '{}.{}.bin'.format(prefix, target_suffix))
        required_files.append(source_file)
        required_files.append(target_file)

    vocab_file = os.path.join(data_path, 'vocab.bin')
    required_files.append(vocab_file)

    absent_file_flag = False
    for filename in required_files:
        if not os.path.isfile(filename):
            absent_file_flag = True
            break
    return not absent_file_flag


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
