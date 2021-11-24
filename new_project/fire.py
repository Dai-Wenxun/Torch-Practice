import argparse

from logging import getLogger

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# from trainer import Trainer
from logger import init_logger
from utils import data_preparation, init_seed
from processors import PROCESSORS


def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface")

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--max_length", default=None, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--do_test', action='store_true',
                        help="Whether to perform test")

    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")

    parser.add_argument("--state", default='info', type=str,
                        help="Logging state")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    init_seed(args.seed)
    init_logger(args)
    logger = getLogger()
    logger.info(args)

    args.device = torch.device('cuda')

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)
    train_data, valid_data, test_data = data_preparation(args, tokenizer)


if __name__ == '__main__':
    main()
