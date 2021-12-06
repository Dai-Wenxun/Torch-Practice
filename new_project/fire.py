import os
import torch
import argparse
from logging import getLogger

from logger import init_logger
from tasks import load_examples, PROCESSORS, TRAIN_SET, DEV_SET, METRICS, DEFAULT_METRICS
from trainer import Trainer, METHODS
from utils import beautify, get_local_time
from modeling import train_single_model


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="The training method to use.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--max_length", default=None, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_adaptation", action='store_true',
                        help='Whether performed domain adaptation')

    # dataset parameters
    parser.add_argument("--train_examples", default=0.1, type=float,
                        help="<= 1 means the ratio to total train examples, > 1 means the number of train examples.")
    parser.add_argument("--dev_examples", default=1.0, type=float,
                        help="<= 1 means the ratio to total train examples, > 1 means the number of train examples.")

    # training & evaluation parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help="Early stopping steps")
    parser.add_argument('--repetitions', default=3, type=int,
                        help="The number of times to repeat training and testing with different seeds.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=list, default=[42, 84, 126],
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.do_adaptation:
        args.output_dir = os.path.join(*args.model_name_or_path.split('/')[:-1], args.method)
    else:
        args.output_dir = os.path.join('./output', args.task_name, args.method, args.model_name_or_path.split('/')[-1])

    # Init logger
    init_logger(args.output_dir)
    logger = getLogger()

    # Prepare task
    processor = PROCESSORS[args.task_name]()
    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=args.train_examples)
    eval_data = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    # Parameters addition
    args.label_list = processor.get_labels()
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()

    if args.method.endswith('mlm'):
        args.train_type = 'mlm_type'
    else:
        args.train_type = 'seq_cls_type'

    logger.info("Parameters: {}".format(beautify(args)))

    trainer = Trainer(args)

    train_single_model(trainer, train_data, eval_data=eval_data)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
