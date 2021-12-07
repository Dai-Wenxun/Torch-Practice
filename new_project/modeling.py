import os
from typing import List
from logging import getLogger

from trainer import Trainer
from utils import set_seed
from tasks import InputExample


logger = getLogger()


def train_single_model(trainer: Trainer, train_data: List[InputExample], eval_data: List[InputExample] = None):
    args = trainer.args
    results = []

    for repetition in range(args.repetitions):
        set_seed(args.seed[repetition])
        logger.info(f'Repetition: {repetition} Seed: {args.seed[repetition]}')

        results.append(trainer.train(train_data, eval_data=eval_data))

    logger.info(results)
    avg_scores = {metric: 0. for metric in args.metrics}
    for rp in range(args.repetitions):
        for metric in args.metrics:
            avg_scores[metric] += results[rp]['scores'][metric] / float(args.repetitions)

    logger.info([f"avg_{metric}': {round(avg_scores[metric], 3)}" for metric in args.metrics])



