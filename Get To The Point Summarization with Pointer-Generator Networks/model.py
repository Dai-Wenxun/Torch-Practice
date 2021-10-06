import torch
import torch.nn as nn

from module import Encoder, Decoder
from utils import greedy_search


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
        self.alignment_method = config['alignment_method']
        self.strategy = config['decoding_strategy']

        if self.strategy == 'beam_search':
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size
        self.vocab_size = data.vocab_size
        self.padding_token_idx = data.padding_token_idx
        self.unknown_token_idx = data.unknown_token_idx
        self.sos_token_idx = data.sos_token_idx
        self.eos_token_idx = data.eos_token_idx
        self.idx2token = data.idx2token
        self.max_target_length = data.max_target_length

        self.source_token_embedder = nn.Embedding(self.vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.encoder = Encoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers,
            self.dropout_ratio, self.bidirectional
        )

        self.decoder = Decoder(
            self.vocab_size, self.embedding_size, self.hidden_size, self.context_size,
            self.num_dec_layers, self.dropout_ratio, self.alignment_method
        )

    def show_example(self, test_data):
        example = test_data.get_example()

        result = []
        oovs = example['oovs_list']
        source_text = example['source_text']
        target_text = example['target_text']
        source_idx = example['source_idx']
        source_length = example['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        encoder_masks = torch.ne(source_idx, self.padding_token_idx)
        extra_zeros = example['extra_zeros']
        extended_source_idx = example['extended_source_idx']

        generated_tokens = []

        input_target_idx = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
        context = torch.zeros((1, 1, self.context_size)).to(self.device)

        for gen_id in range(len(target_text)):
            input_embeddings = self.target_token_embedder(input_target_idx)

            vocab_dists, context, decoder_hidden_states, attn_dist, p_gen = self.decoder(
                input_embeddings, context, encoder_hidden_states, encoder_outputs, encoder_masks,
                extra_zeros, extended_source_idx
            )

            token = None
            if self.strategy == "greedy_search":
                token_idx = greedy_search(vocab_dists)
                if token_idx >= self.vocab_size:
                    token = oovs[token_idx-self.vocab_size]
                    generated_tokens.append(token)
                    token_idx = self.unknown_token_idx
                else:
                    token = self.idx2token[token_idx]
                    generated_tokens.append(token)
                input_target_idx = torch.LongTensor([[token_idx]]).to(self.device)

            expected_token = target_text[gen_id]
            attn_dist = attn_dist.view(-1).tolist()
            token_attn = {word: round(attn, 4) for word, attn in zip(source_text, attn_dist)}
            result.append({'expected': expected_token,
                           'attn': token_attn,
                           'p_copy': round(1.-p_gen.view(-1).item(), 4)})

        result.append({'generated': generated_tokens})

        return result

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

            for gen_id in range(self.max_target_length):
                input_embeddings = self.target_token_embedder(input_target_idx)

                vocab_dists, context, decoder_hidden_states, _, _ = self.decoder(
                    input_embeddings, context, decoder_hidden_states, encoder_output, encoder_mask,
                    extra_zero, bid_extended_source_idx
                )

                if self.strategy == "greedy_search":
                    token_idx = greedy_search(vocab_dists)
                    if token_idx >= self.vocab_size:
                        generated_tokens.append(oovs[bid][token_idx-self.vocab_size])
                        token_idx = self.unknown_token_idx
                    else:
                        generated_tokens.append(self.idx2token[token_idx])
                    input_target_idx = torch.LongTensor([[token_idx]]).to(self.device)

                if self.strategy == 'greedy_search':
                    if token_idx == self.eos_token_idx:
                        break

            generated_corpus.append(generated_tokens[:-1])

        return generated_corpus

    def forward(self, corpus):
        # Encoder
        source_idx = corpus['source_idx']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        # Decoder
        input_target_idx = corpus['input_target_idx']
        input_embeddings = self.target_token_embedder(input_target_idx)  # B x dec_len x 128
        context = torch.zeros((source_idx.size(0), 1, self.context_size)).to(source_idx.device)  # B x 1 x 256
        encoder_masks = torch.ne(source_idx, self.padding_token_idx)
        extra_zeros = corpus['extra_zeros']
        extended_source_idx = corpus['extended_source_idx']

        vocab_dists, _, _, _, _ = self.decoder(
            input_embeddings, context, encoder_hidden_states, encoder_outputs, encoder_masks,
            extra_zeros, extended_source_idx
        )

        # Loss
        output_target_idx = corpus['output_target_idx']
        probs_masks = torch.ne(output_target_idx, self.padding_token_idx)

        gold_probs = torch.gather(vocab_dists, 2, output_target_idx.unsqueeze(2)).squeeze(2)  # B x dec_len

        loss = -torch.log(gold_probs + 1e-12) * probs_masks
        length = corpus['target_length']
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
