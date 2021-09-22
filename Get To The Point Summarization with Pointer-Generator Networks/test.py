import torch
import torch.utils.data as data

import config
from model import Seq2Seq
from data import Vocab, miDataset, data_process

device = torch.device(config.device)

if __name__ == '__main__':
    tokens = 'abcdefghijklmnopqrstuvwxyz'
    source = ['name', 'sex', 'address', 'job', 'interest']
    target = ['dwx', 'male', 'hunan', 'student', 'learning']

    vocab = Vocab(tokens)

    # data_loader
    src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask = data_process(source, target, vocab)
    train_dataset = miDataset(src_tensor, tgt_tensor, src_lens, tgt_lens, tgt_mask)
    data_loader = data.DataLoader(train_dataset, batch_size=config.test_batch_size)

    # model
    model = Seq2Seq().to(device)
    model.load_model()

    model.eval()
    for batch in data_loader:
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)

        model.translate(vocab, batch)
