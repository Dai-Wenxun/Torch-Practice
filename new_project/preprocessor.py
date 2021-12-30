from typing import Union
from abc import ABC, abstractmethod

from tasks import InputExample, InputFeatures, OUTPUT_MODES
from pvp import PVPS, SSt2PromptPVP


class Preprocessor(ABC):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    @abstractmethod
    def get_input_features(self, example: InputExample) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass

    def raw_process(self, example: InputExample):
        inputs = self.tokenizer(
            example.text_a if example.text_a else None,
            example.text_b if example.text_b else None,
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True
        )
        return inputs

    def pad_mask(self, input_ids, token_type_ids):
        attention_mask = [1] * len(input_ids)
        padding_length = self.args.max_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.args.max_length
        assert len(attention_mask) == self.args.max_length
        assert len(token_type_ids) == self.args.max_length

        return input_ids, attention_mask, token_type_ids

    def get_label(self, example: InputExample) -> Union[int, float]:
        output_mode = OUTPUT_MODES[self.args.task_name]
        label_map = {label: i for i, label in enumerate(self.args.label_list)}
        if example.label is None:
            return -100
        elif output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)


class PromptPreprocessor(Preprocessor):
    def __init__(self, args, tokenizer):
        super(PromptPreprocessor, self).__init__(args, tokenizer)
        self.pvp0 = SSt2PromptPVP(self.args, self.tokenizer, 0)
        self.pvp1 = SSt2PromptPVP(self.args, self.tokenizer, 1)

    def get_input_features(self, example: InputExample):
        input_ids, token_type_ids = self.pvp0.encode(example)
        input_ids, attention_mask, token_type_ids = self.pad_mask(input_ids, token_type_ids)
        mlm_labels = self.pvp0.get_mask_positions(input_ids)

        f1 = InputFeatures(input_ids, attention_mask, token_type_ids, mlm_labels=mlm_labels)

        input_ids, token_type_ids = self.pvp1.encode(example)
        input_ids, attention_mask, token_type_ids = self.pad_mask(input_ids, token_type_ids)
        mlm_labels = self.pvp1.get_mask_positions(input_ids)

        f2 = InputFeatures(input_ids, attention_mask, token_type_ids, mlm_labels=mlm_labels)

        return f1, f2


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample) -> InputFeatures:
        inputs = self.raw_process(example)
        input_ids, token_type_ids, attention_mask = \
            inputs["input_ids"], inputs['token_type_ids'], inputs['attention_mask']
        label = self.get_label(example)

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label)


class MLMAdaptPreprocessor(Preprocessor):
    def __init__(self, args, tokenizer):
        super(MLMAdaptPreprocessor, self).__init__(args, tokenizer)

    def get_input_features(self, example: InputExample) -> InputFeatures:
        inputs = self.raw_process(example)
        return InputFeatures(**inputs)


class PromptMLMAdaptPreprocessor(Preprocessor):
    def __init__(self, args, tokenizer):
        super(PromptMLMAdaptPreprocessor, self).__init__(args, tokenizer)
        self.pvp = PVPS[args.task_name](self.args, self.tokenizer, args.pattern_id)

    def get_input_features(self, example: InputExample) -> InputFeatures:
        input_ids, token_type_ids = self.pvp.encode(example)
        input_ids, attention_mask, token_type_ids = self.pad_mask(input_ids, token_type_ids)

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


class MLMPreprocessor(Preprocessor):
    def __init__(self, args, tokenizer):
        super(MLMPreprocessor, self).__init__(args, tokenizer)
        self.pvp = PVPS[args.task_name](self.args, self.tokenizer, args.pattern_id)

    def get_input_features(self, example: InputExample) -> InputFeatures:
        input_ids, token_type_ids = self.pvp.encode(example)
        input_ids, attention_mask, token_type_ids = self.pad_mask(input_ids, token_type_ids)

        label = self.get_label(example)
        mlm_labels = self.pvp.get_mask_positions(input_ids)

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels)
