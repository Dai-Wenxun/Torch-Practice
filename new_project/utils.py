import torch
import random
import datetime
import numpy as np

from torch.utils.data import DataLoader

from data_utils import data_process


def data_preparation(args, tokenizer):
    train_data = None
    if args.do_train:
        train_dataset = data_process(args, tokenizer=tokenizer, set_type='train')
        train_data = DataLoader(
            dataset=train_dataset,
            batch_size=args.per_gpu_train_batch_size,
            shuffle=True,
            drop_last=True
        )

    valid_data = None
    if args.do_eval:
        valid_dataset = data_process(args, tokenizer=tokenizer, set_type='valid')
        valid_data = DataLoader(
            dataset=valid_dataset,
            batch_size=args.per_gpu_eval_batch_size,
            shuffle=True,
            drop_last=False
        )

    test_data = None
    if args.do_test:
        test_dataset = data_process(args, tokenizer=tokenizer, set_type='test')
        test_data = DataLoader(
            dataset=test_dataset,
            batch_size=args.per_gpu_eval_batch_size,
            shuffle=False,
            drop_last=False
        )

    return train_data, valid_data, test_data


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

