import os
import json
import torch
import numpy as np
from tqdm import tqdm
from logging import getLogger
from typing import List, Dict
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import spearmanr, pearsonr

from tasks import InputFeatures, DictDataset
from utils import early_stopping, distillation_loss
from preprocessor import SequenceClassifierPreprocessor

logger = getLogger()

SEQ_CLS_TYPE = 'seq_cls_type'
MLM_TYPE = 'mlm_type'

SEQ_CLS = 'seq_cls'
NON_PROMPT_MLM = 'non_prp_mlm'
HARD_PROMPT_MLM = 'hrd_prp_mlm'
SOFT_PROMPT_MLM = 'sft_prp_mlm'
HYBRID_PROMPT_MLM = 'hybrid_prp_mlm'

METHODS = [SEQ_CLS, NON_PROMPT_MLM, SOFT_PROMPT_MLM, HYBRID_PROMPT_MLM]

PREPROCESSORS = {
    SEQ_CLS_TYPE: SequenceClassifierPreprocessor,
    # MLM_TYPE: MLMPreprocessor,
}

EVALUATION_STEP_FUNCTIONS = {
    SEQ_CLS_TYPE: lambda trainer: trainer.seq_cls_eval_step,
    MLM_TYPE: lambda trainer: trainer.mlm_eval_step
}

TRAIN_STEP_FUNCTIONS = {
    SEQ_CLS_TYPE: lambda trainer: trainer.seq_cls_train_step,
    MLM_TYPE: lambda trainer: trainer.mlm_train_step
}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)
        # self.writer = SummaryWriter(os.path.join(self.args.output_dir, 'runs'))
        self.preprocessor = PREPROCESSORS[self.args.train_type](self.tokenizer, self.args)
        self.train_step = TRAIN_STEP_FUNCTIONS[self.args.train_type](self)
        self.eval_step = EVALUATION_STEP_FUNCTIONS[self.args.train_type](self)

    def train(self, train_data: List[InputExample], eval_data: List[InputExample] = None,
              checkpoint_path: str = None) -> Dict:
        self._init_model(checkpoint_path)
        train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // \
                                         (max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulatin_steps * self.args.num_train_epochs

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

        grad_acc_step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_res = {}

        best_score = -1.0
        stopping_step = 0
        stop_flag, update_flag = False, False

        self.model.zero_grad()

        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(self.args.device) for k, t in batch.items()}

                loss = self.train_step(batch)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (grad_acc_step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        logger.info(json.dumps(logs))

                        scores = self.eval(eval_data)['scores']
                        logger.info(json.dumps(scores))
                        # self.writer.add_scalars('metrics', scores, global_step)
                        valid_score = scores[self.args.metrics[0]]
                        if self.args.stopping_steps > 0:
                            best_score, stopping_step, stop_flag, update_flag = early_stopping(
                                valid_score, best_score, stopping_step, max_step=self.args.stopping_steps)
                        else:
                            update_flag = True

                        if update_flag:
                            best_res = {'global_step': global_step, 'scores': scores}
                            self._save()

                        if stop_flag or 0 < self.args.max_steps <= global_step:
                            logger.info(best_res)
                            epoch_iterator.close()
                            break

                grad_acc_step += 1
            if stop_flag or 0 < self.args.max_steps <= global_step:
                break

        self.model = None
        torch.cuda.empty_cache()

        return best_res

    def eval(self, eval_data: List[InputExample]) -> Dict:
        self.model.to(self.args.device)

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        results = {}
        preds, out_label_ids = None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(self.args.device) for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                logits = self.eval_step(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        results['logits'] = preds
        results['labels'] = out_label_ids
        results['predictions'] = np.argmax(results['logits'], axis=1)

        scores = {}
        for metric in self.args.metrics:
            if metric == 'acc':
                scores[metric] = accuracy_score(results['labels'], results['predictions'])
            elif metric == 'mathws':
                scores[metric] = matthews_corrcoef(results['labels'], results['predictions'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], results['predictions'])
            elif metric == 'prson':
                scores[metric] = pearsonr(results['labels'], results['logits'].reshape(-1))[0]
            elif metric == 'sprman':
                scores[metric] = spearmanr(results['labels'], results['logits'].reshape(-1))[0]
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        results['scores'] = scores

        return results

    def mlm_train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    #     inputs = self._generate_default_inputs(train_batch)
    #     labels = train_batch['labels']
    #
    #     if adaptation:
    #         outputs = self.model(**inputs, labels=labels)
    #         return outputs[0]  # loss
    #
    #     mlm_labels = train_batch['mlm_labels']
    #     outputs = self.model(**inputs, labels=labels)
    #     prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
    #     loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
    #
    #     return loss
    #     #
    #     #
    #     # loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.args.label_list)), labels.view(-1))

    def seq_cls_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False) -> torch.Tensor:
        """Perform a sequence classifier training step."""
        inputs = self._generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']
        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, self.args.temperature)
        else:
            return outputs[0]

    def mlm_eval_step(self):
        pass

    def seq_cls_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self._generate_default_inputs(batch)
        return self.model(**inputs)[0]

    @staticmethod
    def _generate_default_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'],
                  'token_type_ids': batch['token_type_ids']}
        return inputs

    def _generate_dataset(self, examples: List[InputExample]):
        features = self._convert_examples_to_features(examples)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features]),  # might be float
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
        }

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.preprocessor.get_input_features(example)
            features.append(input_features)
            # if ex_index < 5:
            #     logger.info(f'--- Example {ex_index} ---')
            #     logger.info(input_features.pretty_print(self.tokenizer))
        return features

    def _save(self) -> None:
        saved_path = os.path.join(self.args.output_dir, 'finetuned_model')
        # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # model_to_save.save_pretrained(saved_path)
        # self.tokenizer.save_pretrained(saved_path)
        logger.info(f"Model saved at {saved_path}")

    def _init_model(self, checkpoint_path=None):
        if checkpoint_path:
            model_name_or_path = checkpoint_path
        else:
            model_name_or_path = self.args.model_name_or_path

        if self.args.train_type == SEQ_CLS_TYPE:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=len(self.args.label_list)).to(self.args.device)
        elif self.args.train_type == MLM_TYPE:
            self.model = BertForMaskedLM.from_pretrained(model_name_or_path).to(self.args.device)
        logger.info(f'Load parameters from {self.args.model_name_or_path}')
