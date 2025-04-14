import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if the validation metric does not improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth', mode='min'):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
            verbose (bool): If True, prints a message for each validation metric improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            mode (str): One of "min" or "max". In "min" mode, training will stop when the metric stops decreasing;
                        in "max" mode, training will stop when the metric stops increasing.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.early_stop = False
        self.counter = 0
        self.mode = mode

        # Set initial best value depending on the mode.
        if self.mode == 'min':
            self.best_score = np.inf
        elif self.mode == 'max':
            self.best_score = -np.inf
        else:
            raise ValueError("mode must be either 'min' or 'max'")

    def __call__(self, val_metric, model):
        score = val_metric
        if self.mode == 'min':
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:  # mode == 'max'
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        """Saves model when the validation metric improves."""
        if self.verbose:
            print(f'Validation metric improved. Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)

    def clean_up(self):
        """Cleanup if needed."""
        pass
