import math
import os
import torch
import pickle
import random
from logging import getLogger

from enum_type import SpecialTokens


class Dataloader:
    def __init__(self, name, config, dataset, batch_size, shuffle=False, drop_last=True):
        self.name = name
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.device = config['device']
        self.interface_only = config['interface_only']
        if self.name == 'train':
            self.iters_per_epoch = config['iters_per_epoch']

        self.step = batch_size
        self.pr = 0
        self.std_pr = 0
        self.pr_end = len(self.target_text)

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        return None

    def __len__(self):
        if self.name == 'train':
            return self.iters_per_epoch
        else:
            return math.floor(self.pr_end / self.batch_size) if self.drop_last \
                else math.ceil(self.pr_end / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if (self.drop_last and self.std_pr + self.batch_size >= self.pr_end) or \
                (not self.drop_last and self.pr >= self.pr_end):
            self.pr = 0
            self.std_pr = 0
            raise StopIteration()

        if self.name == 'train':
            if self.std_pr == self.iters_per_epoch * self.batch_size:  # 3200
                self.pr = 0
                self.std_pr = 0
                raise StopIteration()

        next_batch = self._next_batch_data()
        self.pr += self.batch_size
        self.std_pr += self.batch_size
        return next_batch


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
        self.target_text_data[:], self.input_target_text_idx_data[:], self.output_target_text_idx_data[:], \
        self.target_idx_length_data[:], self.extended_source_text_idx_data[:], self.oovs_list[:] = zip(*temp)



    def get_reference(self):
        return self.target_text_data

    def _next_batch_data(self):
        source_text = self.source_text_data[self.pr:self.pr + self.step]
        tp_source_text_idx_data = self.source_text_idx_data[self.pr:self.pr + self.step]
        tp_source_idx_length_data = self.source_idx_length_data[self.pr:self.pr + self.step]
        source_idx, source_length = self._pad_batch_sequence(tp_source_text_idx_data, tp_source_idx_length_data)

        target_text = self.target_text_data[self.pr:self.pr + self.step]
        tp_input_target_text_idx_data = self.input_target_text_idx_data[self.pr:self.pr + self.step]
        tp_output_target_text_idx_data = self.output_target_text_idx_data[self.pr:self.pr + self.step]

        tp_target_idx_length_data = self.target_idx_length_data[self.pr:self.pr + self.step]
        input_target_idx, target_length = self._pad_batch_sequence(tp_input_target_text_idx_data,
                                                                   tp_target_idx_length_data)
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

    def get_example(self, sentence):

        source_text = sentence.strip().lower().split()

        source_idx_ = [self.token2idx.get(w, self.unknown_token_idx) for w in source_text]
        source_idx = torch.LongTensor([source_idx_])
        source_length = torch.LongTensor([len(source_idx_)])

        extended_source_idx_, oovs = self._article2ids(source_text)
        extend_source_idx = torch.LongTensor([extended_source_idx_])
        oovs_list = [oovs]
        extra_zeros = self._get_extra_zeros(oovs_list)

        example = {
            'source_idx': source_idx.to(self.device),  # 1 x src_len
            'source_length': source_length.to(self.device),  # 1
            'extended_source_idx': extend_source_idx.to(self.device),  # 1 x src_len
            'extra_zeros': extra_zeros.to(self.device),  # 1 x max_oovs_num
            "oovs_list": oovs_list
        }
        return example
