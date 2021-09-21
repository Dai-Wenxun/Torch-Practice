import torch.utils.data as data
from torch.optim.adagrad import Adagrad

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

    for epoch in range(0, config.max_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            loss = model(batch)

            print("epoch:{}, loss:{}".format(epoch, loss.item()))

            loss.backward()
            optimizer.step()

    model.save_model(optimizer.state_dict())
