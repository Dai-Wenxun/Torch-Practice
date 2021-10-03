import os
import yaml
import torch


class Config(object):
    def __init__(self):
        self._load_yamls()
        self._init_device()

    def _load_yamls(self):
        self.config_dict = {}
        current_path = os.path.dirname(__file__)
        overall_config_file = os.path.join(current_path, 'Yamls/overall.yaml')
        model_config_file = os.path.join(current_path, 'Yamls/model.yaml')
        dataset_config_file = os.path.join(current_path, 'Yamls/dataset.yaml')

        for file in [overall_config_file, model_config_file, dataset_config_file]:
            with open(file, 'r', encoding='utf-8') as f:
                self.config_dict.update(yaml.load(f.read(), Loader=yaml.FullLoader))

    def _init_device(self):
        if self.config_dict['use_gpu']:
            gpu_id = self.config_dict['gpu_id']
            self.config_dict['device'] = torch.device(f'cuda:{gpu_id}'
                                                      if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            return None


