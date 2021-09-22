import os.path
import time

import torch
import torch.nn as nn

import config
from module import Encoder, Decoder, ReduceState


class Seq2Seq(nn.Module):
    def __init__(self, ):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder()
        self.reducer = ReduceState()
        self.decoder = Decoder()

    def forward(self, batch):
        src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask = batch

        encoder_outputs, encoder_features, hidden = self.encoder(src_tensor, src_lens)

        s_t_1 = self.reducer(hidden)

        step_losses = []

        for i in range(0, tgt_lens.max()-1):
            y_t_1 = tgt_tensor[:, i]
            final_dist, s_t_1 = self.decoder(y_t_1, s_t_1)

            gold_probs = torch.gather(final_dist, 1, tgt_tensor[:, i + 1].unsqueeze(1))
            step_loss = -torch.log(gold_probs + config.eps)
            step_loss = step_loss * tgt_mask[:, i + 1].unsqueeze(1)
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / tgt_lens.unsqueeze(1)
        loss = torch.mean(batch_avg_loss)

        return loss

    def save_model(self, optimizer_state_dict):
        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'reducer_state_dict': self.reducer.state_dict(),
            'optimizer': optimizer_state_dict
        }

        model_save_path = os.path.join("./model/model_%s" % time.strftime("%m_%d_%H_%M", time.localtime()))
        if not os.path.exists("./model"):
            os.mkdir("./model")

        torch.save(state, model_save_path)

    def translate(self, vocab):
        pass
