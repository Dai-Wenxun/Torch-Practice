
MLM_ADAPT = 'mlm_adapt'
PROMPT_ADAPT = 'prompt_adapt'

ADAPT_METHODS = [MLM_ADAPT, PROMPT_ADAPT]


class AdaptConfig:
    method = None
    data_dir = None
    output_dir = None
    model_name_or_path = None
    task_name = None
    max_length = None
    train_examples = None
    seed = None
    device = None
    n_gpu = None
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

        if self.method == PROMPT_ADAPT:
            self.n_gpu = 1
