import torch

import torch.nn.functional as F


def greedy_search(vocab_dist):
    return vocab_dist.view(-1).argmax().item()


class Beam_Search_Hypothesis:
    def __init__(self, beam_size, sos_token_idx, eos_token_idx, unknown_token_idx, device, idx2token):
        self.beam_size = beam_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.unknown_token_idx = unknown_token_idx
        self.device = device
        self.idx2token = idx2token
        self.vocab_size = len(idx2token)

        self.hypothetic_token_idx = [[sos_token_idx]]
        self.hypothetic_token = [[idx2token[sos_token_idx]]]
        self.completed_hypotheses = []
        self.hyp_scores = torch.zeros(1).to(device)


    def step(self, gen_idx, vocab_dists, decoder_hidden_states, kwargs=None):

        vocab_dists = torch.log(vocab_dists.squeeze(1)) # hyp_num x vocab_size
        vocab_size = vocab_dists.size(-1)

        live_hyp_num = self.beam_size - len(self.completed_hypotheses)

        tmp_hyp_scores = (self.hyp_scores.unsqueeze(1).expand_as(vocab_dists) + vocab_dists).view(-1)

        top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)

        hyp_ids = (top_pos / vocab_size).long()
        word_ids = top_pos % vocab_size

        new_hypotheses = []
        new_ids = []
        new_scores = []

        for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
            if word_id >= self.vocab_size:
                word_id = self.unknown_token_idx
            new_hyp = self.hypothetic_token_idx[hyp_id] + [word_id]
            if word_id == self.eos_token_idx:
                self.completed_hypotheses.append((new_hyp[1:-1], score / (gen_idx - 1)))
            else:
                new_hypotheses.append(new_hyp)
                new_ids.append(hyp_id)
                new_scores.append(score)

        if len(self.completed_hypotheses) == self.beam_size:
            pass

        self.hypothetic_token_idx = new_hypotheses
        self.hyp_scores = torch.tensor(new_scores).to(self.device)

        hyp_num = len(self.hypothetic_token_idx)

        input_seq = [hyp[-1] for hyp in self.hypothetic_token_idx]
        input_seq = torch.tensor(input_seq).unsqueeze(1).to(self.device)

        returns = [input_seq]
