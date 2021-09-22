import time

import torch.utils.data as data
from torch.optim.adagrad import Adagrad
from torch.utils.tensorboard import SummaryWriter

import config
from model import Seq2Seq
from data import Vocab, miDataset, data_process

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
    model = Seq2Seq()

    # optimizer
    optimizer = Adagrad(model.parameters(), lr=config.lr,
                        initial_accumulator_value=config.adagrad_init_acc)
    # logger
    writer = SummaryWriter("./runs/%s" % time.strftime("%m_%d_%H_%M", time.localtime()))

    iter_id = 0
    while iter_id < config.max_iterations:
        for batch in data_loader:
            optimizer.zero_grad()
            loss = model(batch)
            iter_id += 1
            if iter_id % config.log_iterations == 0:
                writer.add_scalar("loss", loss.item(), iter_id)
                print("iter:{}, loss:{}".format(iter_id, loss.item()))

            loss.backward()
            optimizer.step()

    model.save_model(optimizer.state_dict())

