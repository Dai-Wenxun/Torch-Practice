import os
import csv
import copy
import random
import json
from abc import ABC, abstractmethod
from collections import Counter
from logging import getLogger
from typing import List, Optional, Union

from torch.utils.data import Dataset


logger = getLogger()


class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'\ninput_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'label             = {self.label}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


TASK_NUM_LABELS = {
    "cola": 2,
    # "mnli": 3,
    "mrpc": 2,
    # "sst-2": 2,
    # "sts-b": 1,
    # "qqp": 2,
    # "qnli": 2,
    "rte": 2,
    # "wnli": 2,
}

OUTPUT_MODES = {
    "cola": "classification",
    # "mnli": "classification",
    # "mnli-mm": "classification",
    "mrpc": "classification",
    # "sst-2": "classification",
    # "sts-b": "regression",
    # "qqp": "classification",
    # "qnli": "classification",
    "rte": "classification",
    # "wnli": "classification",
}

PROCESSORS = {
    "cola": ColaProcessor,
    # "mnli": MnliProcessor,
    # "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    # "sst-2": Sst2Processor,
    # "sts-b": StsbProcessor,
    # "qqp": QqpProcessor,
    # "qnli": QnliProcessor,
    "rte": RteProcessor,
    # "wnli": WnliProcessor,
}

METRICS = {
    "cola": ["acc", "mathws"],
    "mrpc": ["acc", 'f1']
}

DEFAULT_METRICS = ["acc"]


TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET]


def load_examples(task, data_dir: str, set_type: str, num_examples: int, seed: int = 42) -> List[InputExample]:
    processor = PROCESSORS[task]()

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    examples = _shuffle_and_restrict(examples, num_examples, seed)

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"{len(examples)} {set_type} examples with label dist: {list(label_distribution.items())}")

    return examples


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples