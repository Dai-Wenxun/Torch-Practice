# from tasks import load_examples
#
# examples = load_examples('rte', './data/rte', 'train', -1)
#
# from transformers import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased')
#
# batch_encoding = tokenizer(
#     [(example.text_a, example.text_b) for example in examples]
# )
# max_len = 0
# min_len = 1e9
# sum_len = 0
# for ids in batch_encoding['input_ids']:
#     if len(ids) > max_len:
#         max_len = len(ids)
#     if len(ids) < min_len:
#         min_len = len(ids)
#     sum_len += len(ids)
#
# avg_len = sum_len / len(batch_encoding['input_ids'])
#
# print(f'max_len: {max_len}, \nmin_len: {min_len}, \navg_len: {avg_len} \n')
#
# import torch
# import torch.nn as nn
#
# func = nn.BCELoss()
# logits = torch.tensor([[1], [1]], dtype=torch.float32)
# labels = torch.tensor([0, 1], dtype=torch.int64)
# loss = func(logits.view(-1), labels)



class AdaptConfig:
    per_gpu_train_batch_size = 64
    gradient_accumulation_steps = 1
    max_steps = -1
    num_train_epochs = 5
    logging_steps = 50
    warmup_steps = 0
    learning_rate = 5e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    mask_ratio = 0.15
    temperature = 0.05

    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def test(a=1, b=1, c=1):
    config = AdaptConfig(locals())

    return config

test()