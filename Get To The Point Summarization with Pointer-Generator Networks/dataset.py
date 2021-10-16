import os
import torch
from logging import getLogger
from data_utils import load_data, build_vocab, text2idx
from enum_type import SpecialTokens


class Dataset:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.max_vocab_size = config['max_vocab_size']
        self.source_max_length = config['src_len']
        self.target_max_length = config['tgt_len']
        self.pointer_gen = config['pointer_gen']

        self.logger = getLogger()
        self._init_special_token()
        self._get_preset()
        self.restored_exist = self._detect_restored()
        if self.restored_exist:
            self._from_restored()
        else:
            self._from_scratch()
        self._info()

    def _get_preset(self):
        for prefix in ['train', 'valid', 'test']:
            setattr(self, f'{prefix}_data', dict())
        self.source_text = []
        self.target_text = []

    def _init_special_token(self):
        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS
        self.padding_token_idx = 0
        self.unknown_token_idx = 1
        self.sos_token_idx = 2
        self.eos_token_idx = 3
        self.special_token_list = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token]

    def _from_scratch(self):
        self._load_data()
        self._build_vocab()
        self._build_data()
        self._dump_data()

    def _from_restored(self):
        self._load_restored()

    def _load_data(self):
        self.logger.info('Loading data from scratch')
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(self.dataset_path, f'{prefix}.src')
            target_file = os.path.join(self.dataset_path, f'{prefix}.tgt')

            source_text = load_data(source_file, self.source_max_length)
            target_text = load_data(target_file, self.target_max_length)

            self.source_text.append(source_text)
            self.target_text.append(target_text)
        self.logger.info('Load finished')

    def _build_vocab(self):
        self.logger.info('Building vocab')
        text_data = self.source_text + self.target_text
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(
            text_data, self.max_vocab_size, self.special_token_list
        )
        self.logger.info('Build finished')

    def _build_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            data_dict = text2idx(self.source_text[i], self.target_text[i], self.token2idx, self.pointer_gen)
            for key, value in data_dict:
                getattr(self, f'{prefix}_data')[key] = value
            getattr(self, f'{prefix}_data')['source_text'] = self.source_text[i]
            getattr(self, f'{prefix}_data')['target_text'] = self.target_text[i]

    def _dump_data(self):
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            data = getattr(self, f'{prefix}_data')
            torch.save(data, filename)

        vocab_file = os.path.join(self.dataset_path, 'vocab.bin')
        torch.save([self.idx2token, self.token2idx, self.max_vocab_size], vocab_file)

    def _detect_restored(self):
        absent_file_flag = False
        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            if not os.path.isfile(filename):
                absent_file_flag = True
                break
        return not absent_file_flag

    def _load_restored(self):
        self.logger.info('Loading data from restored')

        for prefix in ['train', 'valid', 'test']:
            filename = os.path.join(self.dataset_path, f'{prefix}.bin')
            data = torch.load(filename)
            setattr(self, f'{prefix}_data', data)

        vocab_file = os.path.join(self.dataset_path, 'vocab.bin')
        self.idx2token, self.token2idx, self.max_vocab_size = torch.load(vocab_file)

        self.logger.info("Restore finished")

    def _info(self):
        info_str = ''
        self.logger.info(f"Vocab size: {self.max_vocab_size}")
        for prefix in ['train', 'valid', 'test']:
            data = getattr(self, f'{prefix}_data')['target_text']
            info_str += f'{prefix}: {len(data)} cases, '
        self.logger.info(info_str[:-2] + '\n')
