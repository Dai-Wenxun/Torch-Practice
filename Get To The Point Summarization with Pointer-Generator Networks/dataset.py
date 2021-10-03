import os

from data_utils import detect_restored, load_data, build_vocab, dump_data, load_restored


class SpecialTokens:
    PAD = "<|pad|>"
    UNK = "<|unk|>"
    SOS = "<|startoftext|>"
    EOS = "<|endoftext|>"


class Dataset:
    def __init__(self, config):
        self.config = config
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

    def _from_scratch(self, ):
        self._load_data()
        self._build_vocab()
        self._dump_data()

    def _load_data(self):
        for prefix in ['train', 'dev', 'test']:
            source_file = os.path.join(self.data_path, '{}.{}'.format(prefix, self.source_suffix))
            target_file = os.path.join(self.data_path, '{}.{}'.format(prefix, self.target_suffix))

            source_text = load_data(source_file, self.max_source_length)
            target_text = load_data(target_file, self.max_target_length)

            self.source_text_data.append(source_text)
            self.target_text_data.append(target_text)

    def _dump_data(self):
        dump_data(
            self.data_path, self.source_text_data, self.target_text_data, self.idx2token, self.token2idx,
            self.source_suffix, self.target_suffix
        )

    def _build_vocab(self):
        text_data = self.source_text_data + self.target_text_data
        self.idx2token, self.token2idx, self.max_vocab_size = build_vocab(
            text_data, self.max_vocab_size, self.special_tokens
        )

    def _detect_restored(self):
        return detect_restored(self.data_path, self.source_suffix, self.target_suffix)

    def _from_restored(self):
        self.source_text_data, self.target_text_data, self.idx2token, self.token2idx = load_restored(
            self.data_path, self.source_suffix, self.target_suffix
        )
        self.max_vocab_size = len(self.idx2token)
