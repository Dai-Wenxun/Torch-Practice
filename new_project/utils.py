import torch
import random
import numpy as np


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beautify(args):
    args_info = '\n'
    args_info += f"{'data_dir'}={args.data_dir}\n"
    args_info += f"{'model_name_or_path'}={args.model_name_or_path}\n"
    args_info += f"{'task_name'}={args.task_name}\n"
    args_info += f"{'max_length'}={args.max_length}\n"
    args_info += f"{'train_examples'}={args.train_examples}\n"
    args_info += f"{'dev_examples'}={args.dev_examples}\n"
    args_info += f"{'per_gpu_train_batch_size'}={args.per_gpu_train_batch_size}\n"
    args_info += f"{'per_gpu_eval_batch_size'}={args.per_gpu_eval_batch_size}\n"
    args_info += f"{'num_train_epochs'}={args.num_train_epochs}\n"
    args_info += f"{'gradient_accumulation_steps'}={args.gradient_accumulation_steps}\n"
    args_info += f"{'max_steps'}={args.max_steps}\n"
    args_info += f"{'logging_steps'}={args.logging_steps}\n"
    args_info += f"{'repetitions'}={args.repetitions}\n"
    args_info += f"{'warmup_steps'}={args.warmup_steps}\n"
    args_info += f"{'learning_rate'}={args.learning_rate}\n"
    args_info += f"{'weight_decay'}={args.weight_decay}\n"
    args_info += f"{'adam_epsilon'}={args.adam_epsilon}\n"
    args_info += f"{'max_grad_norm'}={args.max_grad_norm}\n"
    args_info += f"{'temperature'}={args.temperature}\n"

    return args_info
