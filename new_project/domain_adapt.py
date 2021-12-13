import os
import torch
import json
import pickle
from tqdm import tqdm
from typing import Tuple, List
from logging import getLogger
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks import DictDataset, load_examples, TRAIN_SET, InputExample
from utils import set_seed

logger = getLogger()


def _print_hyper_params(args):
    args_info = '\n'
    args_info += f"per_gpu_train_batch_size={args['per_gpu_train_batch_size']}\n"
    args_info += f"n_gpu={args['n_gpu']}\n"
    args_info += f"gradient_accumulation_steps={args['gradient_accumulation_steps']}\n"
    args_info += f"max_steps={args['max_steps']}\n"
    args_info += f"num_train_epochs={args['num_train_epochs']}\n"
    args_info += f"warmup_steps={args['warmup_steps']}\n"
    args_info += f"learning_rate={args['learning_rate']}\n"
    args_info += f"weight_decay={args['weight_decay']}\n"
    args_info += f"adam_epsilon={args['adam_epsilon']}\n"
    args_info += f"max_grad_norm={args['max_grad_norm']}\n"
    args_info += f"mask_ratio={args['mask_ratio']}\n"

    return args_info


def _get_special_tokens_mask(tokenizer, token_ids_0):
    return list(map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] else 0,
                    token_ids_0))


def _mask_tokens(input_ids, tokenizer: BertTokenizer, mask_ratio):
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, mask_ratio)
    # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
    #                        labels.tolist()]
    special_tokens_mask = [_get_special_tokens_mask(tokenizer, val) for val in labels.tolist()]
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


def _generate_dataset(examples: List[InputExample], max_length: int, tokenizer: BertTokenizer) -> DictDataset:
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )

    return DictDataset(**batch_encoding)


def API_Adapt(data_dir,
              output_dir,
              model_name_or_path,
              task_name,
              max_length,
              train_examples,
              seed,
              per_gpu_train_batch_size=8,
              gradient_accumulation_steps=1,
              max_steps=-1,
              num_train_epochs=1,
              logging_steps=50,
              warmup_steps=0,
              learning_rate=5e-5,
              weight_decay=0.01,
              adam_epsilon=1e-8,
              max_grad_norm=1.0,
              mask_ratio=0.15
              ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()
    logger.info(_print_hyper_params(locals()))
    set_seed(seed)

    ########################################
    #           TRAINING START            #
    ########################################
    model = BertForMaskedLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    writer = SummaryWriter(output_dir)

    examples, fine_tune_examples = load_examples(task_name, data_dir, TRAIN_SET,
                                                 num_examples=train_examples, seed=seed)

    train_dataset = _generate_dataset(examples, max_length, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // \
                           (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    grad_acc_step = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    for epoch in range(num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
        for _, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: t.to(device) for k, t in batch.items()}
            batch['input_ids'], batch['labels'] = _mask_tokens(batch['input_ids'], tokenizer, mask_ratio)
            loss = model(**batch)[0]

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (grad_acc_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    logs['step'] = global_step
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    logger.info(json.dumps(logs))
                    writer.add_scalar('loss', loss_scalar, global_step)
                    if 0 < max_steps <= global_step:
                        epoch_iterator.close()
                        break
            grad_acc_step += 1
        if 0 < max_steps <= global_step:
            break

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved at {output_dir}")
    with open(os.path.join(output_dir, 'examples.bin'), 'wb') as f:
        pickle.dump(fine_tune_examples, f)

    # Clear cache
    model = None
    torch.cuda.empty_cache()

    return fine_tune_examples

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     # required parameters
#     parser.add_argument("--data_dir", default=None, type=str, required=True,
#                         help="The input data dir. Should contain the data files for the task.")
#     parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
#                         help="Path to the pre-trained model or shortcut name")
#     parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
#                         help="The name of the task to train/evaluate on")
#     parser.add_argument("--max_length", default=None, type=int, required=True,
#                         help="The maximum total input sequence length after tokenization. Sequences longer "
#                              "than this will be truncated, sequences shorter will be padded.")
#
#     # dataset parameters
#     parser.add_argument("--train_examples", default=-1, type=int,
#                         help="The total number of train examples to use, where -1 equals all examples.")
#
#     # training & evaluation parameters
#     parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
#                         help="Batch size per GPU/CPU for training.")
#     parser.add_argument("--num_train_epochs", default=3, type=float,
#                         help="Total number of training epochs.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: Override num_train_epochs.")
#     parser.add_argument('--logging_steps', type=int, default=50,
#                         help="Log every X updates steps.")
#     parser.add_argument('--stopping_steps', type=int, default=10,
#                         help="Early stopping steps")
#     parser.add_argument("--warmup_steps", default=0, type=int,
#                         help="Linear warmup over warmup_steps.")
#     parser.add_argument("--learning_rate", default=1e-5, type=float,
#                         help="The initial learning rate for Adam.")
#     parser.add_argument("--weight_decay", default=0.01, type=float,
#                         help="Weight decay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
#     parser.add_argument('--seed', type=list, default=42,
#                         help="random seed for initialization")
#
#     args = parser.parse_args()
#
#     # Logger Setup
#     output_dir = os.path.join('./output', task_name,
#                                    f"adapted-{model_name_or_path.split('/')[-1]}", f'{get_local_time()}')
#     init_logger(output_dir)
#     logger = getLogger()
#
#     # Gpu Setup
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     n_gpu = torch.cuda.device_count()
#
#     logger.info(_print_hyper_params(args))
#
#     train(args)
