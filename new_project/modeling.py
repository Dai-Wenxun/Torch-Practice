import os
from typing import List
from logging import getLogger

from trainer import Trainer
from utils import set_seed
from tasks import InputExample, load_examples, DEV_SET
from domain_adapt import API_Adapt

logger = getLogger()


def logger_helper(results, metrics, repetitions):
    for res in results:
        logger.info(res)
    avg_scores = {metric: 0. for metric in metrics}
    for rp in range(repetitions):
        for metric in metrics:
            avg_scores[metric] += results[rp]['scores'][metric] / float(repetitions)

    logger.info([f"avg_{metric}': {round(avg_scores[metric], 3)}" for metric in metrics])


def train_single_model(trainer: Trainer):
    args = trainer.args
    fine_tune_results = []
    fine_tune_with_adapted_results = []

    eval_data, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        checkpoint_path = os.path.join(args.output_dir, f'Seed-{domain_seed}')

        logger.info('\nDomain adaptation start:')
        train_data = API_Adapt(args.data_dir, checkpoint_path, args.model_name_or_path,
                               args.task_name, args.max_length, 1.0-args.train_examples, domain_seed)

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)
            logger.info(f'Domain:{domain_repetition}, Seed:{domain_seed}'
                        f' Finetune:{fine_tune_repetition}-Seed:{fine_tune_seed}')

            logger.info("Fine tune without adaptation start: ")
            fine_tune_results.append(trainer.train(train_data, eval_data=eval_data))
            logger.info("Fine tune with adaptation start: ")
            fine_tune_with_adapted_results.append(trainer.train(train_data, eval_data=eval_data,
                                                                checkpoint_path=checkpoint_path))

    logger.info('Fine tune without adaptation results:')
    logger_helper(fine_tune_results, args.metrics, args.repetitions)
    logger.info('Fine tune with adaptation results:')
    logger_helper(fine_tune_with_adapted_results, args.metrics, args.repetitions)
