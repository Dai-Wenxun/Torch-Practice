import torch.nn as nn

from module import Encoder


class Model(nn.Module):
    def __init__(self, config, data):
        super(Model, self).__init__()
        self.batch_size = config['train_batch_size']
        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.attention_type = config['attention_type']
        self.alignment_method = config['alignment_method']
        self.strategy = config['decoding_strategy']

        if self.strategy == 'beam_search':
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size
        self.vocab_size = data.vocab_size
        self.padding_token_idx = data.padding_token_idx
        self.sos_token_idx = data.sos_token_idx
        self.eos_token_idx = data.eos_token_idx

        self.source_token_embedder = nn.Embedding(self.vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.encoder = Encoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers,
            self.dropout_ratio, self.bidirectional
        )

    def forward(self, corpus):
        source_idx = corpus['source_idx']
        source_length = corpus['source_length']

        source_embeddings = self.source_token_embedder(source_idx)
        encoder_outputs, encoder_hidden_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]

        encoder_hidden_states = (encoder_hidden_states[0][::2], encoder_hidden_states[1][::2])





