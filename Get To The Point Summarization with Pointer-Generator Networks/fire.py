from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed

from utils import data_preparation

if __name__ == '__main__':
    config = Config(config_dict={'test_only': False,
                                 'load_experiment': None,
                                 'is_coverage': False})

    init_seed(config['seed'], config['reproducibility'])
    train_data, valid_data, test_data = data_preparation(config)
    model = Model(config, train_data).to(config['device'])
    trainer = Trainer(config, model)

    if config['test_only']:
        test_result = trainer.evaluate(test_data, load_best_model=True, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, test_data)
        print('best valid loss: {}, best valid ppl: {}'.format(best_valid_score, best_valid_result))
        test_result = trainer.evaluate(test_data, load_best_model=True)

    print('test result: {}'.format(test_result))
