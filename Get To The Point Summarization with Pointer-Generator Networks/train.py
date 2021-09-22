import time
import numpy as np
import torch
import torch.utils.data as data
from torch.optim.adagrad import Adagrad
from torch.utils.tensorboard import SummaryWriter

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
    data_loader = data.DataLoader(train_dataset, batch_size=config.batch_size)

    # model
    model = Seq2Seq().to(device)

    # optimizer
    optimizer = Adagrad(model.parameters(), lr=config.lr,
                        initial_accumulator_value=config.adagrad_init_acc)
    # logger
    writer = SummaryWriter("./runs/%s" % time.strftime("%m_%d_%H_%M", time.localtime()))

    for epoch in range(1, config.max_epochs+1):
        epoch_loss = []
        for batch in data_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)

            loss = model(batch)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.log_epochs == 0:
            writer.add_scalar("loss", np.average(epoch_loss), epoch)
            print("epoch:{}, loss:{}".format(epoch, np.average(epoch_loss)))

    model.save_model()
