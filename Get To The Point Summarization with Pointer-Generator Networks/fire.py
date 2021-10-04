
from config import Config
from dataset import Dataset
from dataloader import Dataloader
from utils import init_seed


if __name__ == '__main__':
    # fire()
    config = Config()

    init_seed(config['seed'], config['reproducibility'])

    datatset = Dataset(config)

    train_dataset, _, _ = datatset.build()

    train_data = Dataloader(
        config=config,
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
        drop_last=False
    )

