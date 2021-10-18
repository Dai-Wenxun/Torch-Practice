import torch
import torch.nn as nn

from module import Encoder, Decoder
from strategy import greedy_search, Beam_Search_Hypothesis


class Model(nn.Module):
    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.device = config['device']
        self.max_vocab_size = dataset.max_vocab_size
        self.padding_token_idx = dataset.padding_token_idx
        self.unknown_token_idx = dataset.unknown_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx
        self.max_target_length = dataset.target_max_length
        self.id2token = dataset.idx2token

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']

        self.is_attention = config['is_attention']
        self.is_pgen = config['is_pgen']
        self.is_coverage = config['is_coverage']

        if self.is_coverage:
            self.cov_loss_lambda = config['cov_loss_lambda']

        if self.strategy == 'beam_search':
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size

        self.source_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.encoder = Encoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers,
            self.dropout_ratio, self.bidirectional
        )

        self.decoder = Decoder(
            self.max_vocab_size, self.embedding_size, self.hidden_size,
            self.num_dec_layers, self.dropout_ratio,
            is_attention=self.is_attention,  is_pgen=self.is_pgen, is_coverage=self.is_coverage
        )

    def encode(self, source_idx, source_length):
        source_embeddings = self.source_token_embedder(source_idx)
        encoder_outputs, encoder_hidden_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]

        encoder_hidden_states = (encoder_hidden_states[0][::2], encoder_hidden_states[1][::2])

        return encoder_outputs, encoder_hidden_states

    def generate(self, corpus):
        generated_corpus = []

        oovs = corpus['oovs_list']

        source_idx = corpus['source_idx']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        batch_size = len(source_idx)
        src_len = len(source_idx[0])

        for bid in range(batch_size):
            generated_tokens = []

            input_target_idx = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            decoder_hidden_states = (encoder_hidden_states[0][:, bid, :].unsqueeze(1),
                                     encoder_hidden_states[1][:, bid, :].unsqueeze(1))

            kwargs = {}
            if self.is_attention:
                kwargs['encoder_outputs'] = encoder_outputs[bid, :, :].unsqueeze(0)
                kwargs['encoder_masks'] = torch.ne(source_idx[bid], self.padding_token_idx).unsqueeze(0).to(self.device)
                kwargs['context'] = torch.zeros((1, 1, self.context_size)).to(self.device)

            if self.is_pgen:
                kwargs['extra_zeros'] = corpus['extra_zeros'][bid, :].unsqueeze(0)
                kwargs['extended_source_idx'] = corpus['extended_source_idx'][bid, :].unsqueeze(0)

            if self.is_coverage:
                kwargs['coverages'] = torch.zeros((1, 1, src_len)).to(self.device)

            if self.strategy == 'beam_search':
                hypothesis = Beam_Search_Hypothesis(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.unknown_token_idx,
                    self.device, self.id2token
                )

            for gen_id in range(self.max_target_length):
                input_embeddings = self.target_token_embedder(input_target_idx)

                vocab_dists, decoder_hidden_states, kwargs = self.decoder(
                    input_embeddings, decoder_hidden_states, kwargs=kwargs
                )

                if self.strategy == 'greedy_search':
                    token_idx = greedy_search(vocab_dists)
                elif self.strategy == 'beam_search':
                    input_target_idx, decoder_hidden_states, kwargs = hypothesis.step(
                        gen_id, vocab_dists, decoder_hidden_states, kwargs)


                if self.strategy == 'greedy_search':
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        if oovs is not None:
                            if token_idx >= self.vocab_size:
                                generated_tokens.append(oovs[bid][token_idx - self.vocab_size])
                                token_idx = self.unknown_token_idx
                        else:
                            generated_tokens.append(self.idx2token[token_idx])
                        input_target_idx = torch.LongTensor([[token_idx]]).to(self.device)
                elif self.strategy == 'beam_search':
                    if hypothesis.stop():
                        break
            if self.strategy == 'beam_search':
                generated_tokens = hypothesis.generate()

            generated_corpus.append(generated_tokens)

        return generated_corpus

    def forward(self, corpus):
        # Encoder
        source_idx = corpus['source_idx']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        batch_size = len(source_idx)
        src_len = len(source_idx[0])

        # Decoder
        input_target_idx = corpus['input_target_idx']
        input_embeddings = self.target_token_embedder(input_target_idx)  # B x dec_len x 128

        kwargs = {}
        if self.is_attention:
            kwargs['encoder_outputs'] = encoder_outputs  # B x src_len x 256
            kwargs['encoder_masks'] = torch.ne(source_idx, self.padding_token_idx).to(self.device)  # B x src_len
            kwargs['context'] = torch.zeros((batch_size, 1, self.context_size)).to(self.device)  # B x 1 x 256

        if self.is_pgen:
            kwargs['extra_zeros'] = corpus['extra_zeros']  # B x max_oovs_num
            kwargs['extended_source_idx'] = corpus['extended_source_idx']  # B x src_len

        if self.is_coverage:
            kwargs['coverages'] = torch.zeros((batch_size, 1, src_len)).to(self.device)  # B x 1 x src_len

        vocab_dists, decoder_hidden_states, kwargs = self.decoder(
            input_embeddings, encoder_hidden_states, kwargs=kwargs
        )
        # Loss
        output_target_idx = corpus['output_target_idx']
        probs_masks = torch.ne(output_target_idx, self.padding_token_idx)

        gold_probs = torch.gather(vocab_dists, 2, output_target_idx.unsqueeze(2)).squeeze(2)  # B x dec_len
        nll_loss = -torch.log(gold_probs + 1e-12)
        if self.is_coverage:
            coverage_loss = torch.sum(torch.min(kwargs['attn_dists'], kwargs['coverages']), dim=2)  # B x dec_len
            nll_loss = nll_loss + self.cov_loss_lambda * coverage_loss

        loss = nll_loss * probs_masks
        length = corpus['target_length']
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
