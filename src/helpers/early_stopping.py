import os

import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_val_acc = 0
        self.early_stop = False
        self.verbose = verbose
        self.path = path

    def __call__(self, val_acc, model):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f'Saved best model with val_acc = {self.best_val_acc:.4f}')

    def clean_up(self):
        if self.early_stop:
            os.remove(self.path)
            if self.verbose:
                print(f'Removed model checkpoint at {self.path}')
