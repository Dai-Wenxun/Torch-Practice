import os
import pickle
from logging import getLogger

from numpy import mean, std

from trainer import Trainer
from utils import set_seed
from tasks import load_examples, DEV_SET, TRAIN_SET
from domain_adapt import AdaptTrainer

logger = getLogger()


def logger_helper(results, metrics):
    for res in results:
        logger.info(res)
    avg_scores = {metric: [] for metric in metrics}
    for rp in range(len(results)):
        for metric in metrics:
            avg_scores[metric].append(results[rp]['scores'][metric] * 100)

    logger.info([f"avg_{metric}': {round(mean(avg_scores[metric]), 3)}, "
                 f"std_{metric}: {round(std(avg_scores[metric]), 2)}" for metric in metrics])


def train_single_model(trainer: Trainer, adapt_only=False, pretrained_path="X"):
    args = trainer.args
    fine_tune_results = []
    fine_tune_with_adapted_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        adapt_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                           num_examples=1. - args.train_examples, seed=domain_seed)
        if not pretrained_path:
            checkpoint_path = os.path.join(args.output_dir, f'Seed-{domain_seed}')
            logger.info(f'\n{args.adapt_method} domain adaptation start:')
            AdaptTrainer(args.adapt_method, args.data_dir, checkpoint_path, args.model_name_or_path,
                         args.task_name, args.max_length, domain_seed, args.device, args.n_gpu).train(adapt_examples)
        else:
            checkpoint_path = os.path.join(pretrained_path, f'Seed-{domain_seed}')

        if not adapt_only:
            for fine_tune_repetition in range(args.repetitions):
                fine_tune_seed = args.seed[fine_tune_repetition]
                set_seed(fine_tune_seed)

                info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
                logger.info(info)

                # logger.info("Fine tune with adaptation start: ")
                # args.saved_path = os.path.join(args.output_dir, info, 'adapted')
                # fine_tune_with_adapted_results.append(trainer.train(fine_tune_examples, eval_examples=eval_examples,
                #                                                     checkpoint_path=checkpoint_path))
                logger.info("Fine tune without adaptation start: ")
                args.saved_path = os.path.join(args.output_dir, info, 'non-adapted')
                fine_tune_results.append(trainer.train(fine_tune_examples, eval_examples=eval_examples))

    if not adapt_only:
        logger.info('Fine tune without adaptation results:')
        logger_helper(fine_tune_results, args.metrics)
        logger.info('Fine tune with adaptation results:')
        logger_helper(fine_tune_with_adapted_results, args.metrics)
