from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed

from utils import data_preparation

if __name__ == '__main__':
    config = Config()
    init_seed(config['seed'], config['reproducibility'])
    train_data, valid_data, test_data = data_preparation(config)
    model = Model(config, train_data).to(config['device'])

