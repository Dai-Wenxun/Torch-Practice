import torch
import torch.nn as nn

from module import Encoder, Decoder
from strategy import greedy_search, Beam_Search_Hypothesis


class Model(nn.Module):
    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.device = config['device']
        self.dataset = dataset

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.attention = config['attention']
        self.strategy = config['decoding_strategy']
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
            self.vocab_size, self.embedding_size, self.hidden_size, self.context_size,
            self.num_dec_layers, self.dropout_ratio, self.alignment_method, self.is_coverage
        )

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            value = getattr(self.dataset, name)
            if value is not None:
                return value
        return super().__getattr__(name)

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

        # Encoder
        source_idx = corpus['source_idx']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        # Decoder
        encoder_masks = torch.ne(source_idx, self.padding_token_idx)
        extra_zeros = corpus['extra_zeros']
        extended_source_idx = corpus['extended_source_idx']

        for bid in range(source_idx.size(0)):
            generated_tokens = []

            input_target_idx = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            context = torch.zeros((1, 1, self.context_size)).to(self.device)
            decoder_hidden_states = (encoder_hidden_states[0][:, bid, :].unsqueeze(1),
                                     encoder_hidden_states[1][:, bid, :].unsqueeze(1))
            encoder_output = encoder_outputs[bid, :, :].unsqueeze(0)
            encoder_mask = encoder_masks[bid, :].unsqueeze(0)
            extra_zero = extra_zeros[bid, :].unsqueeze(0)
            bid_extended_source_idx = extended_source_idx[bid, :].unsqueeze(0)
            coverage = None
            if self.is_coverage:
                coverage = torch.zeros((1, 1, len(source_idx[0]))).to(self.device)

            for gen_id in range(self.max_target_length):
                input_embeddings = self.target_token_embedder(input_target_idx)

                vocab_dists, context, decoder_hidden_states, _, _, coverage = self.decoder(
                    input_embeddings, context, decoder_hidden_states, encoder_output, encoder_mask,
                    extra_zero, bid_extended_source_idx, coverage
                )

                if self.strategy == "greedy_search":
                    token_idx = greedy_search(vocab_dists)

                if self.strategy == 'greedy_search':
                    if token_idx == self.eos_token_idx:
                        break
                    else:
                        if token_idx >= self.vocab_size:
                            generated_tokens.append(oovs[bid][token_idx - self.vocab_size])
                            token_idx = self.unknown_token_idx
                        else:
                            generated_tokens.append(self.idx2token[token_idx])
                        input_target_idx = torch.LongTensor([[token_idx]]).to(self.device)

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
        context = torch.zeros((batch_size, 1, self.context_size)).to(self.device)  # B x 1 x 256
        encoder_masks = torch.ne(source_idx, self.padding_token_idx)
        extra_zeros = corpus['extra_zeros']
        extended_source_idx = corpus['extended_source_idx']

        coverage = None
        if self.is_coverage:
            coverage = torch.zeros((batch_size, 1, src_len)).to(self.device)

        vocab_dists, _, _, attn_dists, _, coverages = self.decoder(
            input_embeddings, context, encoder_hidden_states, encoder_outputs, encoder_masks,
            extra_zeros, extended_source_idx, coverage
        )
        # Loss
        output_target_idx = corpus['output_target_idx']
        probs_masks = torch.ne(output_target_idx, self.padding_token_idx)

        gold_probs = torch.gather(vocab_dists, 2, output_target_idx.unsqueeze(2)).squeeze(2)  # B x dec_len
        nll_loss = -torch.log(gold_probs + 1e-12)
        if self.is_coverage:
            coverage_loss = torch.sum(torch.min(attn_dists, coverages), dim=2)  # B x dec_len
            nll_loss = nll_loss + self.cov_loss_lambda * coverage_loss

        loss = nll_loss * probs_masks
        length = corpus['target_length']
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
