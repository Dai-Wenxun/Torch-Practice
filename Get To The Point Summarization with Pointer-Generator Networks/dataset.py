import os
from logging import getLogger
from data_utils import detect_restored, load_data, build_vocab, dump_data, load_restored
from enum_type import SpecialTokens


class Dataset:
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()
        self.padding_token = SpecialTokens.PAD
        self.unknown_token = SpecialTokens.UNK
        self.sos_token = SpecialTokens.SOS
        self.eos_token = SpecialTokens.EOS

        self.special_tokens = [self.padding_token, self.unknown_token, self.sos_token, self.eos_token]

        self.data_path = config['data_path']
        self.max_vocab_size = config['max_vocab_size']
        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length']
        self.source_suffix = config['source_suffix'].lower()
        self.target_suffix = config['target_suffix'].lower()

        self.restored_exist = self._detect_restored()

        self._get_preset()
        if self.restored_exist:
            self._from_restored()
        else:
            self._from_scratch()

    def _get_preset(self):
        self.idx2token = {}
        self.token2idx = {}
        self.source_text_data = []
        self.target_text_data = []

    def _from_scratch(self):
        self._load_data()
        self._build_vocab()
        self._dump_data()

    def _load_data(self):
        self.logger.info('Loading data from scratch')
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(self.data_path, '{}.{}'.format(prefix, self.source_suffix))
            target_file = os.path.join(self.data_path, '{}.{}'.format(prefix, self.target_suffix))

            source_text = load_data(source_file, self.max_source_length)
            target_text = load_data(target_file, self.max_target_length)

            self.source_text_data.append(source_text)
            self.target_text_data.append(target_text)
        self.logger.info('Load finished')

    def _dump_data(self):
        self.logger.info('Dumping data')
        dump_data(
            self.data_path, self.source_text_data, self.target_text_data, self.idx2token, self.token2idx,
            self.source_suffix, self.target_suffix
        )
        self.logger.info("Dump finished")

    def _build_vocab(self):
        self.logger.info('Building vocab')
        text_data = self.source_text_data + self.target_text_data
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(
            text_data, self.max_vocab_size, self.special_tokens
        )
        self.logger.info('Build finished')

    def _detect_restored(self):
        return detect_restored(self.data_path, self.source_suffix, self.target_suffix)

    def _from_restored(self):
        self.logger.info('Loading data from restored')

        self.source_text_data, self.target_text_data, self.idx2token, self.token2idx = load_restored(
            self.data_path, self.source_suffix, self.target_suffix
        )
        self.max_vocab_size = len(self.idx2token)

        self.logger.info("Restore finished")

    def build(self):
        info_str = ''
        corpus_list = []
        self.logger.info("Vocab size: {}".format(self.max_vocab_size))
        for i, prefix in enumerate(['train', 'dev', 'test']):
            source_text_data = self.source_text_data[i]
            target_text_data = self.target_text_data[i]
            tp_data = {
                'idx2token': self.idx2token,
                'token2idx': self.token2idx,
                'source_text_data': source_text_data,
                'target_text_data': target_text_data,
                'vocab_size': self.max_vocab_size,
                'max_source_length': self.max_source_length,
                'max_target_length': self.max_target_length,
            }
            corpus_list.append(tp_data)
            info_str += '{}: {} cases, '.format(prefix, len(source_text_data))

        self.logger.info(info_str[:-2] + '\n')
        return corpus_list
