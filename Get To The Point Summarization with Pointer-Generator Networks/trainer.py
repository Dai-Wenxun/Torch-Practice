import os
import numpy as np
import torch.nn.utils
import torch.optim as optim

from time import time
from utils import early_stopping, ensure_dir


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # training settings
        self.epochs = config['epochs']
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.grad_clip = config['grad_clip']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = 1e9
        self.best_valid_result = None
        self.optimizer = self._build_optimizer()

        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = config['filename'] + '.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.generated_text_dir = config['generated_text_dir']
        ensure_dir(self.generated_text_dir)
        saved_text_file = self.config['filename'] + '.txt'
        self.saved_text_file = os.path.join(self.generated_text_dir, saved_text_file)

        self.example_list = []

    def _build_optimizer(self):
        if self.learner == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_epoch(self, train_data):
        self.model.train()
        total_loss = 0.

        for data in train_data:
            self.optimizer.zero_grad()
            loss = self.model(data)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        train_loss = total_loss / len(train_data)
        return train_loss

    def _valid_epoch(self, valid_data):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.

            for data in valid_data:
                loss = self.model(data)
                total_loss += loss.item()

            valid_loss = total_loss / len(valid_data)
            ppl = np.exp(valid_loss)
        return valid_loss, ppl

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def fit(self, train_data, valid_data, test_data):
        if self.start_epoch >= self.epochs or self.epochs <= 0:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            if test_data is not None:
                with torch.no_grad():
                    self.model.eval()
                    self.example_list.append(self.model.show_example(test_data))

            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            training_end_time = time()

            train_loss_output = "epoch %d training [time: %.2fs, train_loss: %.4f]" % \
                                (epoch_idx, training_start_time - training_end_time, train_loss)
            print(train_loss_output)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step
                )
                valid_end_time = time()

                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_loss: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid ppl: {}'.format(valid_result)
                print(valid_score_output)
                print(valid_result_output)

                if update_flag:
                    self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result
                    print('Saving current best: %s' % self.saved_model_file)
                if stop_flag:
                    print('Finished training, best eval result in epoch %d' %
                          (epoch_idx - self.cur_step * self.eval_step))

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, eval=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            print('Loading model structure and parameters from {}'.format(checkpoint_file))

        self.model.eval()

        if not eval:
            generate_sentence = self.model.generate(next(eval_data))
            print(' '.join(generate_sentence[0]))
            return
