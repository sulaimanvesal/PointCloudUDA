import numpy as np
from torch import save
import os


class EarlyStoppingCallback:

    def __init__(self, patience, mode="min"):
        assert mode=="max" or mode=="min", "mode can only be /'min/' or /'max/'"
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_result = np.Inf if mode=='min' else np.NINF

    def step(self, monitor):
        # check whether the current loss is lower than the previous best value.
        better = False
        if self.mode=="max":
            better = monitor > self.best_result
        else:
            better = monitor < self.best_result
        # if not count up for how long there was no progress
        if better:
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.counter >= self.patience


class ModelCheckPointCallback:

    def __init__(self, mode="min", best_model_name=None, save_best=False, entire_model=False,
                 save_last_model=False, model_name="../weights/model_checkpoint.pt", n_epochs=200):
        assert mode=="max" or mode=="min", "mode can only be /'min/' or /'max/'"
        self.mode = mode
        self.best_result = np.Inf if mode=='min' else np.NINF
        self.model_name = model_name
        if best_model_name is None:
            best_model_name = model_name
        self.best_model_name_base, self.ext = os.path.splitext(best_model_name)
        self.best_model_name = best_model_name
        self.entire_model = entire_model
        self.save_last_model = save_last_model
        self.n_epochs = n_epochs
        self.epoch = 0
        self._save_best = save_best

    def step(self, monitor, model, epoch, optimizer=None):
        # check whether the current loss is lower than the previous best value.
        if self._save_best:
            if self.mode=="max":
                better = monitor > self.best_result
            else:
                better = monitor < self.best_result
            opt_to_save = optimizer
            if self.entire_model:
                to_save = model
            else:
                to_save = model.state_dict()
                if optimizer is not None:
                    opt_to_save = optimizer.state_dict()
            if better:
                self.best_result = monitor
                self.epoch = epoch
                save({'epoch': epoch,
                    'model_state_dict': to_save,
                      'optimizer_state_dict': opt_to_save}, self.best_model_name)
            if epoch == self.n_epochs:
                model_name = '{}{}{}{}'.format(self.best_model_name_base, '.Scr', np.around(self.best_result, 3), self.ext)
                os.rename(self.best_model_name, model_name)
        if self.save_last_model and (epoch == self.n_epochs):
            opt_to_save = optimizer
            if self.entire_model:
                to_save = model
            else:
                to_save = model.state_dict()
                if optimizer is not None:
                    opt_to_save = optimizer.state_dict()
            save({'epoch': epoch,
                  'model_state_dict': to_save,
                  'optimizer_state_dict': opt_to_save}, self.model_name)


if __name__ == '__main__':
    import os
    a = '../../train_pointnet.py'
    b, c = os.path.splitext(a)
    print('{}{}{}{}'.format(b, '.Scr', np.around(0.862323232, 2), c))
