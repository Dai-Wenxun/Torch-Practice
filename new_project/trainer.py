import os
import json

import torch
import numpy as np
from tqdm import tqdm
from logging import getLogger
from typing import List, Dict, Union
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, matthews_corrcoef


from tasks import InputFeatures, DictDataset
from tasks import PROCESSORS, OUTPUT_MODES

logger = getLogger('root')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.model_name_or_path, num_labels=len(self.args.label_list)).to(args.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)

    def train_single_model(self, train_data: List[InputExample], eval_data: List[InputExample]) -> dict:
        results_dict = {}
        self.model.to(self.args.device)

        results_dict['before_training_acc'] = self.evaluate(eval_data)['scores']
        global_step, tr_loss = self._train(train_data)
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss
        results_dict['after_training_acc'] = self.evaluate(eval_data)['scores']

        self._save()

        return results_dict

    def evaluate(self, eval_data: List[InputExample]) -> dict:
        self.model.to(self.args.device)

        results = self._eval(eval_data)

        predictions = np.argmax(results['logits'], axis=1)
        scores = {}

        for metric in self.args.metrics:
            if metric == 'acc':
                scores[metric] = accuracy_score(results['labels'], predictions)
            elif metric == 'mathws':
                scores[metric] = matthews_corrcoef(results['labels'], predictions)
            else:
                raise ValueError(f"Metric '{metric}' not implemented")

        results['scores'] = scores
        results['predictions'] = predictions
        return results

    def _save(self) -> None:
        saved_path = os.path.join(self.args.output_dir, 'model')
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(saved_path)
        self.tokenizer.save_pretrained(saved_path)
        logger.info(f"Model saved at {saved_path}")

    def _train(self, train_data: List[InputExample]):
        train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // \
                                (max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(self.args.device) for k, t in batch.items()}
                outputs = self.model(**batch)

                loss = outputs[0]

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        logger.info(json.dumps({**{'step': global_step}, **logs}))

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
                step += 1

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def _eval(self, eval_data: List[InputExample]) -> Dict:
        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        preds, out_label_ids = None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(self.args.device) for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        return {
            'logits': preds,
            'labels': out_label_ids,
        }

    def _generate_dataset(self, data: List[InputExample]):
        features = self._convert_examples_to_features(data)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features]),  # might be float
        }
        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        processor = PROCESSORS[self.args.task_name]()
        label_list = processor.get_labels()
        output_mode = OUTPUT_MODES[self.args.task_name]

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float]:
            if example.label is None:
                return -100
            if output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = self.tokenizer(
            [(example.text_a, example.text_b) for example in examples],
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)
            # if i < 5:
            #     logger.info(f'--- Example {i} ---')
            #     logger.info(feature.pretty_print(self.tokenizer))
        return features

    # def _mask_tokens(self, input_ids):
    #     """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    #     labels = input_ids.clone()
    #     # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    #     probability_matrix = torch.full(labels.shape, 0.15)
    #     special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
    #                            labels.tolist()]
    #     probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    #
    #     masked_indices = torch.bernoulli(probability_matrix).bool()
    #
    #     # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
    #     if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
    #         ignore_value = -100
    #     else:
    #         ignore_value = -1
    #
    #     labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens
    #
    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    #     input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    #
    #     # 10% of the time, we replace masked input tokens with random word
    #     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    #     input_ids[indices_random] = random_words[indices_random]
    #
    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #     return input_ids, labels
    #
    # def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
    #                                    temperature: float = 1, **_) -> torch.Tensor:
    #     """Perform a sequence classifier training step."""
    #
    #     inputs = self.generate_default_inputs(batch)
    #     if not use_logits:
    #         inputs['labels'] = batch['labels']
    #
    #     outputs = self.model(**inputs)
    #
    #     if use_logits:
    #         logits_predicted, logits_target = outputs[0], batch['logits']
    #         return distillation_loss(logits_predicted, logits_target, temperature)
    #     else:
    #         return outputs[0]
    #
    # def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     """Perform a sequence classifier evaluation step."""
    #     inputs = self.generate_default_inputs(batch)
    #     return self.model(**inputs)[0]