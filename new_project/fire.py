import os
import torch
import argparse
from logging import getLogger

from logger import init_logger
from tasks import load_examples, PROCESSORS, TRAIN_SET, DEV_SET, METRICS, DEFAULT_METRICS
from trainer import Trainer
from utils import beautify


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--max_length", default=None, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longebr "
                             "than this will be truncated, sequences shorter will be padded.")

    # dataset parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--dev_examples", default=-1, type=int,
                        help="The total number of dev examples to use, where -1 equals all examples.")

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
    parser.add_argument("--temperature", default=2, type=float,
                        help="")

    # Other optional parameters
    parser.add_argument('--seed', type=list, default=[42, 84, 126],
                        help="random seed for initialization")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    args = parser.parse_args()
    args.output_dir = os.path.join('./output', args.task_name, args.model_name_or_path.split('/')[-1])

    # Init logger
    init_logger(args)
    logger = getLogger()
    logger.info("Parameters: {}".format(beautify(args)))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))

    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=args.train_examples)
    eval_data = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    trainer = Trainer(args)

    results = trainer.train(train_data, eval_data)

    logger.info(results)

    avg_scores = {metric: 0. for metric in args.metrics}
    for rp in range(args.repetitions):
        for metric in args.metrics:
            avg_scores[metric] += results[rp][f'Rp_{rp}_scores'][metric] / float(args.repetitions)

    logger.info([f"avg_{metric}': {round(avg_scores[metric], 3)}" for metric in args.metrics])


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    main()
