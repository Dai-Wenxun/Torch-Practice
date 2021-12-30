import os
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


def train_final_model(args, checkpoint_path='output/sst-2/bert-base-uncased/21-1227-2337'):
    final_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        _, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                              num_examples=1. - args.train_examples, seed=domain_seed)

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)

            # args.learning_rate = args.learning_rate * 0.1
            args.logging_steps = 5
            # args.warmup_steps = 1000
            trainer = Trainer(args)

            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)
            logger.info("final_training start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'final_training')
            pretrained_path = os.path.join(checkpoint_path, info, "fine_tune_with_lm_training_and_mlm_adapted")
            final_results.append(
                trainer.train(fine_tune_examples,
                              eval_examples=eval_examples,
                              checkpoint_path=pretrained_path))

    logger.info('final_results:')
    logger_helper(final_results, args.metrics)


def train_single_model(args, checkpoint_path='output/sst-2/bert-base-uncased'):
    fine_tune_with_lm_training_results = []
    fine_tune_with_lm_training_and_mlm_adapted_results = []
    fine_tune_with_lm_training_and_prompt_mlm_adapted_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                               num_examples=1. - args.train_examples, seed=domain_seed)

        pretrained_path = os.path.join(checkpoint_path, 'MLM_Adapt', f'Seed-{domain_seed}')
        pretrained_path_prompt = os.path.join(checkpoint_path, 'P-MLM_Adapt', f'Seed-{domain_seed}')

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)

            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)

            # logger.info("fine_tune_with_lm_training start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_lm_training')
            # fine_tune_with_lm_training_results.append(
            #     trainer.train(fine_tune_examples,
            #                   eval_examples=eval_examples,
            #                   unlabeled_examples=unlabeled_examples))

            args.method = 'mlm_pretrain'
            trainer = Trainer(args)
            logger.info("stage2_with_mlm_adapted start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'stage2_with_mlm_adapted')
            trainer.pre_train(unlabeled_examples)

            args.method = 'mlm'
            trainer = Trainer(args)
            logger.info("fine_tune_with_lm_training_and_mlm_adapted start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_lm_training_and_mlm_adapted')
            fine_tune_with_lm_training_and_mlm_adapted_results.append(
                trainer.train(fine_tune_examples,
                              eval_examples=eval_examples,
                              checkpoint_path=args.saved_path))

            # logger.info("fine_tune_with_lm_training_and_prompt_mlm_adapted start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_lm_training_and_prompt_mlm_adapted')
            # fine_tune_with_lm_training_and_prompt_mlm_adapted_results.append(
            #     trainer.train(fine_tune_examples, eval_examples=eval_examples,
            #                   unlabeled_examples=unlabeled_examples,
            #                   checkpoint_path=pretrained_path_prompt))

    logger.info('fine_tune_with_lm_training_results:')
    logger_helper(fine_tune_with_lm_training_results, args.metrics)
    logger.info('fine_tune_with_lm_training_and_mlm_adapted_results:')
    logger_helper(fine_tune_with_lm_training_and_mlm_adapted_results, args.metrics)
    logger.info('fine_tune_with_lm_training_and_prompt_mlm_adapted_results:')
    logger_helper(fine_tune_with_lm_training_and_prompt_mlm_adapted_results, args.metrics)


def baseline_train(args, checkpoint_path='output/sst-2/bert-base-uncased'):
    fine_tune_results = []
    fine_tune_with_mlm_adapted_results = []
    fine_tune_with_prompt_mlm_adapted_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                               num_examples=1. - args.train_examples, seed=domain_seed)

        pretrained_path = os.path.join(checkpoint_path, 'MLM_Adapt', f'Seed-{domain_seed}')
        pretrained_path_prompt = os.path.join(checkpoint_path, 'P-MLM_Adapt', f'Seed-{domain_seed}')

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)
            trainer = Trainer(args)
            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)

            # logger.info("fine_tune start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune')
            # fine_tune_results.append(trainer.train(fine_tune_examples, eval_examples=eval_examples))

            logger.info("fine_tune_with_mlm_adapted start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_mlm_adapted')
            fine_tune_with_mlm_adapted_results.append(
                trainer.train(fine_tune_examples,
                              eval_examples=eval_examples,
                              checkpoint_path=pretrained_path))

            logger.info("fine_tune_with_prompt_mlm_adapted start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_prompt_mlm_adapted')
            fine_tune_with_prompt_mlm_adapted_results.append(
                trainer.train(fine_tune_examples,
                              eval_examples=eval_examples,
                              checkpoint_path=pretrained_path_prompt))

    logger.info('fine_tune_results:')
    logger_helper(fine_tune_results, args.metrics)
    logger.info('fine_tune_with_mlm_adapted_results:')
    logger_helper(fine_tune_with_mlm_adapted_results, args.metrics)
    logger.info('fine_tune_with_prompt_mlm_adapted_results:')
    logger_helper(fine_tune_with_prompt_mlm_adapted_results, args.metrics)


def adapt_train(args):
    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                               num_examples=1. - args.train_examples, seed=domain_seed)

        pretrained_path = os.path.join(args.output_dir, 'MLM_Adapt', f'Seed-{domain_seed}')
        AdaptTrainer('mlm_adapt', args.data_dir, pretrained_path, args.model_name_or_path,
                     args.task_name, args.max_length, args.device, args.n_gpu, args.pattern_id).train(
            unlabeled_examples)
        pretrained_path = os.path.join(args.output_dir, 'P-MLM_Adapt', f'Seed-{domain_seed}')
        AdaptTrainer('prompt_mlm_adapt', args.data_dir, pretrained_path, args.model_name_or_path,
                     args.task_name, args.max_length, args.device, args.n_gpu, args.pattern_id).train(
            unlabeled_examples)
