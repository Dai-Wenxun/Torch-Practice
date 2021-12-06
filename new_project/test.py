
from tasks import load_examples

examples = load_examples('rte', './data/rte', 'train', -1)


from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased')

batch_encoding = tokenizer(
    [(example.text_a, example.text_b) for example in examples]
)
max_len = 0
min_len = 1e9
sum_len = 0
for ids in batch_encoding['input_ids']:
    if len(ids) > max_len:
        max_len = len(ids)
    if len(ids) < min_len:
        min_len = len(ids)
    sum_len += len(ids)


avg_len = sum_len / len(batch_encoding['input_ids'])

print(f'max_len: {max_len}, \nmin_len: {min_len}, \navg_len: {avg_len} \n')
