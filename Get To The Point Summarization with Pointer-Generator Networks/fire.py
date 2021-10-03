
from config import Config
from dataset import Dataset
from utils import init_seed, data_preparation


if __name__ == '__main__':
    # fire()
    config = Config()

    init_seed(config['seed'], config['reproducibility'])

    datatset = Dataset(config)

    pass


