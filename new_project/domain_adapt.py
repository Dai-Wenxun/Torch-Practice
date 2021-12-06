import os
import torch
import json
import shutil
import argparse
from tqdm import tqdm
from logging import getLogger
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks import PROCESSORS, DictDataset, load_examples, TRAIN_SET
from logger import init_logger
from utils import get_local_time, early_stopping


def _print_hyper_params(args):
    args_info = '\n'
    args_info += f"per_gpu_train_batch_size={args.per_gpu_train_batch_size}\n"
    args_info += f"n_gpu={args.n_gpu}\n"
    args_info += f"num_train_epochs={args.num_train_epochs}\n"
    args_info += f"gradient_accumulation_steps={args.gradient_accumulation_steps}\n"
    args_info += f"max_steps={args.max_steps}\n"
    args_info += f"warmup_steps={args.warmup_steps}\n"
    args_info += f"learning_rate={args.learning_rate}\n"
    args_info += f"weight_decay={args.weight_decay}\n"
    args_info += f"adam_epsilon={args.adam_epsilon}\n"
    args_info += f"max_grad_norm={args.max_grad_norm}\n"
    args_info += f"seed={args.seed}\n"

    return args_info


def _mask_tokens(input_ids, tokenizer: BertTokenizer):
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    ignore_value = -100

    labels[~masked_indices] = ignore_value

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random].to(labels.device)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels


def _generate_dataset(args, tokenizer: BertTokenizer) -> DictDataset:
    examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                             num_examples=args.train_examples)
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    return DictDataset(**batch_encoding)


def _save_model(args, model, tokenizer):
    saved_path = os.path.join(args.output_dir, 'adapted_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(saved_path)
    tokenizer.save_pretrained(saved_path)
    logger.info(f"Model saved at {saved_path}")


def train(args):
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    writer = SummaryWriter(args.output_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = _generate_dataset(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // \
                                (max(1, len(train_dataloader) // args.gradient_accumulation_steps)) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    grad_acc_step = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    best_score = -1e9
    stopping_step = 0
    stop_flag, update_flag = False, False

    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
        for _, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: t.to(args.device) for k, t in batch.items()}
            batch['input_ids'], batch['labels'] = _mask_tokens(batch['input_ids'], tokenizer)
            loss = model(**batch)[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (grad_acc_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs['step'] = global_step
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    logger.info(json.dumps(logs))
                    writer.add_scalar('loss', loss_scalar, global_step)

                    if args.stopping_steps > 0:
                        best_score, stopping_step, stop_flag, update_flag = early_stopping(
                            -loss_scalar, best_score, stopping_step, max_step=args.stopping_steps)
                    else:
                        update_flag = True

                    if update_flag:
                        _save_model(args, model, tokenizer)

                    if stop_flag or 0 < args.max_steps <= global_step:
                        epoch_iterator.close()
                        break
            grad_acc_step += 1
        if stop_flag or 0 < args.max_steps <= global_step:
            break

    # Clear cache
    model = None
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--max_length", default=None, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    # dataset parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")

    # training & evaluation parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
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
    parser.add_argument('--seed', type=list, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # Logger Setup
    args.output_dir = os.path.join('./output', args.task_name,
                                   f"adapted-{args.model_name_or_path.split('/')[-1]}", f'{get_local_time()}')
    init_logger(args.output_dir)
    logger = getLogger()

    # Gpu Setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()

    logger.info(_print_hyper_params(args))

    train(args)
