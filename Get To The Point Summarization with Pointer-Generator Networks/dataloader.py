import torch
import random
from enum_type import SpecialTokens


class Dataloader:
    def __init__(self, config, dataset, batch_size, shuffle=False, drop_last=True):
        self.config = config
        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.device = config['device']

        self.step = batch_size
        self.pr = 0
        self.std_pr = 0

        self.vocab_size = dataset['vocab_size']
        self.max_source_length = dataset['max_source_length']
        self.max_target_length = dataset['max_target_length']

        self._get_preset()
        self._data_process()

    def _get_preset(self):
        self.source_text_idx_data = []
        self.input_target_text_idx_data = []
        self.output_target_text_idx_data = []
        self.source_idx_length_data = []
        self.target_idx_length_data = []
        self.extended_source_text_idx_data = []
        self.oovs_list = []

    def _data_process(self):
        required_key_list = ['source_text_data', 'target_text_data', 'idx2token', 'token2idx']
        for dataset_attr in required_key_list:
            assert dataset_attr in self.dataset
            setattr(self, dataset_attr, self.dataset[dataset_attr])

        for source_text, target_text in zip(self.source_text_data, self.target_text_data):
            source_text_idx = [self.token2idx.get(w, self.unknown_token_idx) for w in source_text]
            extended_source_text_idx, oovs = self._article2ids(source_text)

            input_target_text_idx = [self.sos_token_idx] + [self.token2idx.get(w, self.unknown_token_idx)
                                                            for w in target_text]

            output_target_text_idx = self._abstract2ids(target_text, oovs) + [self.eos_token_idx]

            self.source_text_idx_data.append(source_text_idx)
            self.input_target_text_idx_data.append(input_target_text_idx)
            self.output_target_text_idx_data.append(output_target_text_idx)
            self.source_idx_length_data.append(len(source_text_idx))
            self.target_idx_length_data.append(len(input_target_text_idx))
            self.extended_source_text_idx_data.append(extended_source_text_idx)
            self.oovs_list.append(oovs)

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

    @property
    def padding_token_idx(self):
        return self.token2idx[self.padding_token]

    @property
    def unknown_token_idx(self):
        return self.token2idx[self.unknown_token]

    @property
    def sos_token_idx(self):
        return self.token2idx[self.sos_token]

    @property
    def eos_token_idx(self):
        return self.token2idx[self.eos_token]

    @property
    def pr_end(self):
        return len(self.input_target_text_idx_data)

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

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def _shuffle(self):
        temp = list(
            zip(
                self.source_text_data, self.source_text_idx_data, self.source_idx_length_data,
                self.target_text_data, self.input_target_text_idx_data, self.output_target_text_idx_data,
                self.target_idx_length_data, self.extended_source_text_idx_data, self.oovs_list
            )
        )
        random.shuffle(temp)
        self.source_text_data[:], self.source_text_idx_data[:], self.source_idx_length_data[:], \
        self.target_text_data[:], self.target_text_idx_data[:], self.target_idx_length_data[:], \
        self.extended_source_text_idx_data[:], self.oovs_list[:] = zip(*temp)

    def __next__(self):
        if (self.drop_last and self.std_pr + self.batch_size >= self.pr_end) or \
            (not self.drop_last and self.pr >= self.pr_end):
            self.pr = 0
            self.std_pr = 0
            raise StopIteration()

        next_batch = self._next_batch_data()
        self.pr += self.batch_size
        self.std_pr += self.batch_size
        return next_batch

    def _next_batch_data(self):
        source_text = self.source_text_data[self.pr:self.pr + self.step]
        tp_source_text_idx_data = self.source_text_idx_data[self.pr:self.pr + self.step]
        tp_source_idx_length_data = self.source_idx_length_data[self.pr:self.pr + self.step]
        source_idx, source_length = self._pad_batch_sequence(tp_source_text_idx_data, tp_source_idx_length_data)

        target_text = self.target_text_data[self.pr:self.pr + self.step]
        tp_input_target_text_idx_data = self.input_target_text_idx_data[self.pr:self.pr + self.step]
        tp_output_target_text_idx_data = self.output_target_text_idx_data[self.pr:self.pr + self.step]

        tp_target_idx_length_data = self.target_idx_length_data[self.pr:self.pr + self.step]
        input_target_idx, target_length = self._pad_batch_sequence(tp_input_target_text_idx_data, tp_target_idx_length_data)
        output_target_idx, _ = self._pad_batch_sequence(tp_output_target_text_idx_data, tp_target_idx_length_data)

        tp_extended_source_text_idx_data = self.extended_source_text_idx_data[self.pr:self.pr + self.step]
        extend_source_idx, _ = self._pad_batch_sequence(tp_extended_source_text_idx_data, tp_source_idx_length_data)

        tp_oovs_list = self.oovs_list[self.pr:self.pr + self.step]
        extra_zeros = self._get_extra_zeros(tp_oovs_list)

        batch_data = {
            'source_text': source_text,
            'source_idx': source_idx.to(self.device),
            'source_length': source_length.to(self.device),
            'target_text': target_text,
            'input_target_idx': input_target_idx.to(self.device),
            'output_target_idx': output_target_idx.to(self.device),
            'target_length': target_length.to(self.device),
            'extended_source_idx': extend_source_idx.to(self.device),
            'extra_zeros': extra_zeros.to(self.device),
            "oovs_list": tp_oovs_list
        }
        return batch_data
