import os
import torch
import json
import pickle
import torch.nn as nn
from abc import ABC
from tqdm import tqdm
from typing import Tuple, List
from logging import getLogger
from transformers import BertForMaskedLM, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks import DictDataset, load_examples, TRAIN_SET, InputExample, InputFeatures
from utils import set_seed

logger = getLogger()


class AdaptTrainer:
    def __init__(self,
                 method,
                 data_dir,
                 output_dir,
                 model_name_or_path,
                 task_name,
                 max_length,
                 train_examples,
                 seed,
                 device, 
                 n_gpu, 
                 per_gpu_train_batch_size=64,
                 gradient_accumulation_steps=1,
                 max_steps=-1,
                 num_train_epochs=5,
                 logging_steps=50,
                 warmup_steps=0,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 adam_epsilon=1e-8,
                 max_grad_norm=1.0,
                 mask_ratio=0.15,
                 temperature=0.05
                 ):
        if method == 'prompt':
            n_gpu = 1
        self.args = locals()
        if method == 'prompt':
            self.model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False).to(device)
        elif method == 'mlm':
            self.model = BertForMaskedLM.from_pretrained(model_name_or_path).to(device)

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.writer = SummaryWriter(output_dir)

    def train(self):
        set_seed(self.args['seed'])
        examples, fine_tune_examples = load_examples(self.args['task_name'], self.args['data_dir'], TRAIN_SET,
                                                     num_examples=self.args['train_examples'], seed=self.args['seed'])

        train_dataset = DatasetGenerator(self.args, self.tokenizer).generate(examples)

        train_sampler = RandomSampler(train_dataset)
        train_batch_size = self.args['per_gpu_train_batch_size'] * max(1, self.args['n_gpu'])
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if self.args['max_steps'] > 0:
            t_total = self.args['max_steps']
            self.args['num_train_epochs'] = self.args['max_steps'] // \
                               (max(1, len(train_dataloader) // self.args['gradient_accumulation_steps'])) + 1
        else:
            t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args['warmup_steps'],
                                                    num_training_steps=t_total)

        # multi-gpu training
        if self.args['n_gpu'] > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        grad_acc_step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.zero_grad()

        for epoch in range(self.args['num_train_epochs']):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(self.args['device']) for k, t in batch.items()}

                loss = None
                if self.args['method'] == 'prompt':
                    loss = self._prompt_train_step(batch)
                elif self.args['method'] == 'mlm':
                    loss = self._mlm_train_step(batch)

                if self.args['n_gpu'] > 1:
                    loss = loss.mean()
                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']
                loss.backward()

                tr_loss += loss.item()
                if (grad_acc_step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args['logging_steps'] > 0 and global_step % self.args['logging_steps'] == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / self.args['logging_steps']
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        logger.info(json.dumps(logs))
                        self.writer.add_scalar('loss', loss_scalar, global_step)
                        if 0 < self.args['max_steps'] <= global_step:
                            epoch_iterator.close()
                            break
                grad_acc_step += 1
            if 0 < self.args['max_steps'] <= global_step:
                break

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args['output_dir'])
        self.tokenizer.save_pretrained(self.args['output_dir'])
        logger.info(f"Model saved at {self.args['output_dir']}")
        with open(os.path.join(self.args['output_dir'], 'examples.bin'), 'wb') as f:
            pickle.dump(fine_tune_examples, f)

        # Clear cache
        self.model = None
        torch.cuda.empty_cache()

        return fine_tune_examples

    def _mlm_train_step(self, batch):
        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'])
        loss = self.model(**batch)[0]
        return loss

    def _prompt_train_step(self, batch):
        inputs_a = {'input_ids': batch['input_ids_a'], 'attention_mask': batch['attention_mask_a'],
                    'token_type_ids': batch['token_type_ids_a']}
        mlm_labels_a = batch['mlm_labels_a']

        logits_a = self.model(**inputs_a)[0][mlm_labels_a >= 0]

        inputs_b = {'input_ids': batch['input_ids_b'], 'attention_mask': batch['attention_mask_b'],
                    'token_type_ids': batch['token_type_ids_b']}
        mlm_labels_b = batch['mlm_labels_b']

        logits_b = self.model(**inputs_b)[0][mlm_labels_b >= 0]

        cos_sim = nn.CosineSimilarity(dim=-1)(logits_a.unsqueeze(1), logits_b.unsqueeze(0)) / self.args['temperature']

        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(logits_a.device)

        loss = loss_fct(cos_sim, labels)

        return loss

    def _mask_tokens(self, input_ids):
        def _get_special_tokens_mask(tokenizer, token_ids_0):
            return list(
                map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] else 0,
                    token_ids_0))

        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, self.args['mask_ratio'])
        # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
        #                        labels.tolist()]
        special_tokens_mask = [_get_special_tokens_mask(self.tokenizer, val) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        ignore_value = -100

        labels[~masked_indices] = ignore_value

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random].to(labels.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def _print_hyper_params(self):
        args_info = '\n'
        args_info += f"method={self.args['method']}\n"
        args_info += f"per_gpu_train_batch_size={self.args['per_gpu_train_batch_size']}\n"
        args_info += f"n_gpu={self.args['n_gpu']}\n"
        args_info += f"gradient_accumulation_steps={self.args['gradient_accumulation_steps']}\n"
        args_info += f"max_steps={self.args['max_steps']}\n"
        args_info += f"num_train_epochs={self.args['num_train_epochs']}\n"
        args_info += f"warmup_steps={self.args['warmup_steps']}\n"
        args_info += f"learning_rate={self.args['learning_rate']}\n"
        args_info += f"weight_decay={self.args['weight_decay']}\n"
        args_info += f"adam_epsilon={self.args['adam_epsilon']}\n"
        args_info += f"max_grad_norm={self.args['max_grad_norm']}\n"
        args_info += f"mask_ratio={self.args['mask_ratio']}\n"
        args_info += f"temperature={self.args['temperature']}\n"
        logger.info(args_info)


class DatasetGenerator(ABC):
    def __init__(self, args, tokenizer: BertTokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.tokenizer.mask_token_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    def encode(self, parts_a, parts_b) -> InputFeatures:
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(self.tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(self.tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        attention_mask = [1] * len(input_ids)
        padding_length = self.args['max_length'] - len(input_ids)

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.args['max_length']
        assert len(attention_mask) == self.args['max_length']
        assert len(token_type_ids) == self.args['max_length']

        mlm_labels = self.get_mask_positions(input_ids)

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             mlm_labels=mlm_labels)

    @staticmethod
    def _seq_length(parts: List[Tuple[List[int], bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[List[int], bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[List[int], bool]], parts_b: List[Tuple[List[int], bool]]):
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - self.args['max_length']

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def get_input_features(self, example: InputExample) -> Tuple[InputFeatures, InputFeatures]:
        if example.text_b is None:
            f_a_parts_a = ['This sentence : "', self.shortenable(example.text_a),
                           '" means ', self.tokenizer.mask_token, '.']
            feature_a = self.encode(f_a_parts_a, None)
            f_b_parts_a = ['This sentence of "', self.shortenable(example.text_a),
                           '" means ', self.tokenizer.mask_token, '.']
            feature_b = self.encode(f_b_parts_a, None)

            return feature_a, feature_b

        else:
            f_a_parts_a, f_a_parts_b = ['This sentence : "', self.shortenable(example.text_a)], \
                                       [self.shortenable(example.text_b), '" means ', self.tokenizer.mask_token, '.']
            feature_a = self.encode(f_a_parts_a, f_a_parts_b)

            f_b_parts_a, f_b_parts_b = ['This sentence of "', self.shortenable(example.text_a)], \
                                       [self.shortenable(example.text_b), '" means ', self.tokenizer.mask_token, '.']
            feature_b = self.encode(f_b_parts_a, f_b_parts_b)

            return feature_a, feature_b

    def generate(self, examples):
        feature_dict = None
        if self.args['method'] == 'mlm':
            feature_dict = self.tokenizer(
                [(example.text_a, example.text_b) for example in examples],
                max_length=self.args['max_length'],
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
        elif self.args['method'] == 'prompt':
            features_a, features_b = [], []
            for ex_index, ex in enumerate(examples):
                feature_a, feature_b = self.get_input_features(ex)

                features_a.append(feature_a)
                features_b.append(feature_b)
                # if ex_index < 5:
                #     logger.info(f'--- Example {ex_index} ---')
                #     logger.info(feature_a.pretty_print(self.tokenizer))
                #     logger.info(feature_b.pretty_print(self.tokenizer))

            feature_dict = {
                'input_ids_a': torch.tensor([f.input_ids for f in features_a], dtype=torch.long),
                'attention_mask_a': torch.tensor([f.attention_mask for f in features_a], dtype=torch.long),
                'token_type_ids_a': torch.tensor([f.token_type_ids for f in features_a], dtype=torch.long),
                'mlm_labels_a': torch.tensor([f.mlm_labels for f in features_a], dtype=torch.long),

                'input_ids_b': torch.tensor([f.input_ids for f in features_b], dtype=torch.long),
                'attention_mask_b': torch.tensor([f.attention_mask for f in features_b], dtype=torch.long),
                'token_type_ids_b': torch.tensor([f.token_type_ids for f in features_b], dtype=torch.long),
                'mlm_labels_b': torch.tensor([f.mlm_labels for f in features_b], dtype=torch.long),
            }

        return DictDataset(**feature_dict)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    AdaptTrainer('prompt', './data/sst-2', './log', './model/bert-base-uncased',
                 'sst-2', 64, 0.99, 100, 'cuda', 4).train()
    #
    # set_seed(self.args['seed'])
    # examples, fine_tune_examples = load_examples(self.args['task_name'], './data/sst-2', TRAIN_SET,
    #                                              num_examples=0.99, seed=100)