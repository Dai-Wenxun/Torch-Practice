import os

import torch
import torch.nn as nn

import config
from module import Encoder, Decoder, ReduceState
from data import Vocab, make_encoder_mask


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.reducer = ReduceState()
        self.decoder = Decoder()

    def forward(self, batch):
        src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask = batch

        encoder_outputs, encoder_features, hidden = self.encoder(src_tensor, src_lens)

        encoder_mask = make_encoder_mask(src_lens)

        s_t_1 = self.reducer(hidden)
        c_t_1 = torch.zeros((encoder_mask.size(0), config.hidden_size * 2), device=src_tensor.device)

        step_losses = []

        for i in range(0, tgt_lens.max() - 1):
            y_t_1 = tgt_tensor[:, i]
            final_dist, s_t_1, c_t_1 = self.decoder(y_t_1, s_t_1, c_t_1,
                                                    encoder_outputs, encoder_features, encoder_mask)

            gold_probs = torch.gather(final_dist, 1, tgt_tensor[:, i + 1].unsqueeze(1))
            step_loss = -torch.log(gold_probs + config.eps)
            step_loss = step_loss * tgt_mask[:, i + 1].unsqueeze(1)
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / tgt_lens.unsqueeze(1)
        loss = torch.mean(batch_avg_loss)

        return loss

    def save_model(self):
        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'reducer_state_dict': self.reducer.state_dict(),
        }
        if not os.path.exists("./model"):
            os.mkdir("./model")
        torch.save(state, config.model_path)

    def load_model(self):
        state = torch.load(config.model_path)
        self.encoder.load_state_dict(state['encoder_state_dict'])
        self.decoder.load_state_dict(state['decoder_state_dict'])
        self.reducer.load_state_dict(state['reducer_state_dict'])

    def translate(self, vocab: Vocab, batch: list):
        src_tensor, _, src_lens, _, _ = batch

        encoder_outputs, encoder_features, hidden = self.encoder(src_tensor, src_lens)
        encoder_pad_mask = make_encoder_mask(src_lens)

        s_t_1 = self.reducer(hidden)

        for b_id in range(len(batch[0])):
            id_list = []
            y_t_1 = torch.tensor([vocab.STAT_id()], dtype=torch.int64, device=src_tensor.device)
            b_s_t_1 = (s_t_1[0][:, b_id, :].unsqueeze(1), s_t_1[1][:, b_id, :].unsqueeze(1))
            b_c_t_1 = torch.zeros((encoder_pad_mask.size(0), config.hidden_size * 2), device=src_tensor.device)
            for i in range(0, config.max_dec_steps):

                final_dist, b_s_t_1, b_c_t_1 = self.decoder(y_t_1, b_s_t_1, b_c_t_1,
                                                            encoder_outputs, encoder_features, encoder_pad_mask)

                y_t_1 = torch.argmax(final_dist.view(-1)).unsqueeze(0)  # greedy search

                if y_t_1.item() == vocab.END_id():
                    break

                id_list.append(y_t_1.item())

            print(vocab.translate(src_tensor[b_id][:src_lens[b_id]].tolist()), "->", vocab.translate(id_list))
