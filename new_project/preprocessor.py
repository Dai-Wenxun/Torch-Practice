from typing import Union
from abc import ABC, abstractmethod

from tasks import InputExample, InputFeatures, OUTPUT_MODES


class Preprocessor(ABC):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.label_map = {label: i for i, label in enumerate(args.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample, adaptation: bool = False) -> InputFeatures:
        inputs = self.tokenizer(
            example.text_a if example.text_a else None,
            example.text_b if example.text_b else None,
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True,
        )
        output_mode = OUTPUT_MODES[self.args.task_name]

        def label_from_example(example: InputExample) -> Union[int, float]:
            if example.label is None:
                return -100
            elif output_mode == "classification":
                return self.label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        input_ids, token_type_ids, attention_mask = \
            inputs["input_ids"], inputs['token_type_ids'], inputs['attention_mask']

        label = label_from_example(example)
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, logits=logits)

# class MLMPreprocessor(Preprocessor):
#     """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""
#
#     def get_input_features(self, example: InputExample) -> InputFeatures:
#
#         if priming:
#             input_ids, token_type_ids = self.pvp.encode(example, priming=True)
#             priming_data = example.meta['priming_data']  # type: List[InputExample]
#
#             priming_input_ids = []
#             for priming_example in priming_data:
#                 pe_input_ids, _ = self.pvp.encode(priming_example, priming=True, labeled=True)
#                 priming_input_ids += pe_input_ids
#
#             input_ids = priming_input_ids + input_ids
#             token_type_ids = self.wrapper.tokenizer.create_token_type_ids_from_sequences(input_ids)
#             input_ids = self.wrapper.tokenizer.build_inputs_with_special_tokens(input_ids)
#         else:
#             input_ids, token_type_ids = self.pvp.encode(example)
#
#         attention_mask = [1] * len(input_ids)
#         padding_length = self.wrapper.config.max_seq_length - len(input_ids)
#
#         if padding_length < 0:
#             raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")
#
#         input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
#         attention_mask = attention_mask + ([0] * padding_length)
#         token_type_ids = token_type_ids + ([0] * padding_length)
#
#         assert len(input_ids) == self.wrapper.config.max_seq_length
#         assert len(attention_mask) == self.wrapper.config.max_seq_length
#         assert len(token_type_ids) == self.wrapper.config.max_seq_length
#
#         label = self.label_map[example.label] if example.label is not None else -100
#         logits = example.logits if example.logits else [-1]
#
#         if labelled:
#             mlm_labels = self.pvp.get_mask_positions(input_ids)
#             if self.wrapper.config.model_type == 'gpt2':
#                 # shift labels to the left by one
#                 mlm_labels.append(mlm_labels.pop(0))
#         else:
#             mlm_labels = [-1] * self.wrapper.config.max_seq_length
#
#         return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                              label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx)