import os
import torch
import random
import datetime
import numpy as np

from dataset import Dataset
from dataloader import Dataloader


def data_preparation(config):
    dataset = Dataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.build()

    test_data = Dataloader(
        name='test',
        config=config,
        dataset=test_dataset,
        batch_size=config['eval_batch_size'],
        drop_last=False
    )

    if config['interface_only']:
        return test_data

    train_data = Dataloader(
        name='train',
        config=config,
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True
    )

    valid_data = Dataloader(
        name='dev',
        config=config,
        dataset=valid_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False
    )

    return train_data, valid_data, test_data


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def init_seed(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def early_stopping(value, best, cur_step, max_step):
    stop_flag = False
    update_flag = False

    if value < best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
    return best, cur_step, stop_flag, update_flag
