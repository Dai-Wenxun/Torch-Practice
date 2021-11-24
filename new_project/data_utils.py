import os
import torch
from logging import getLogger
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
from torch.utils.data import TensorDataset

from processors import InputExample, InputFeatures, PROCESSORS, OUTPUT_MODES


logger = getLogger()


def data_process(args, tokenizer=None, set_type=None):
    data_dir = os.path.join(args.data_dir, args.task_name)
    saved_path = os.path.join(data_dir, f'{set_type}.bin')
    processor = PROCESSORS[args.task_name]()

    if os.path.isfile(saved_path):
        tensors = torch.load(saved_path)
        return TensorDataset(*tensors)

    if set_type == 'train':
        examples = processor.get_train_examples(data_dir)
    elif set_type == 'valid':
        examples = processor.get_dev_examples(data_dir)
    elif set_type == 'test':
        examples = processor.get_test_examples(data_dir)
    else:
        raise ValueError(f"Dataset type: {set_type} undefined")

    features = _convert_examples_to_features(examples, tokenizer, args.max_length, task=args.task_name)
    tensors = _convert_tensors(features)
    torch.save(tensors, saved_path)

    dataset = TensorDataset(*tensors)

    return dataset


def _convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        a list of task-specific``InputFeatures`` which can be fed to the model.

    """
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = PROCESSORS[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = OUTPUT_MODES[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


def _convert_tensors(features):
    input_ids = torch.tensor([f.input_ids for f in features])
    attention_mask = torch.tensor([f.attention_mask for f in features])
    token_type_ids = torch.tensor([f.token_type_ids for f in features])
    label = torch.tensor([f.label for f in features])
    return input_ids, attention_mask, token_type_ids, label
