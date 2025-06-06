import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0, monitor='val_loss', mode='min', verbose=False):
        """
        Early stopping to stop training when validation loss does not improve.

        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change in monitored value to qualify as improvement.
            monitor (str): Metric to monitor (e.g., 'val_loss').
            mode (str): 'min' for decreasing metric, 'max' for increasing metric.
            verbose (bool): If True, prints a message for each early stopping event.
        """
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

        # Set initial comparison value
        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = float('-inf')

    def __call__(self, monitor_value, model):
        """
        Check if the monitored value has improved. If not, increment the counter.
        If it surpasses the patience, set the early_stop flag.

        Args:
            monitor_value (float): Current value of the monitored metric.
            model (torch.nn.Module): The model to save the best weights.
        """
        if self.mode == 'min':
            if monitor_value < self.best - self.delta:
                self.best = monitor_value
                self.counter = 0
                self.best_model_wts = copy.deepcopy(model.state_dict())  # Save best weights
                if self.verbose:
                    print(f"Validation {self.monitor} improved to {self.best:.4f}. Saving best model.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered: {self.monitor} didn't improve for {self.patience} epochs.")
        elif self.mode == 'max':
            if monitor_value > self.best + self.delta:
                self.best = monitor_value
                self.counter = 0
                self.best_model_wts = copy.deepcopy(model.state_dict())  # Save best weights
                if self.verbose:
                    print(f"Validation {self.monitor} improved to {self.best:.4f}. Saving best model.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered: {self.monitor} didn't improve for {self.patience} epochs.")

    def load_best_model(self, model):
        """
        Load the best model weights into the given model.

        Args:
            model (torch.nn.Module): The model to load the weights into.
        """
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)
            if self.verbose:
                print("Best model weights loaded.")
        else:
            if self.verbose:
                print("No best model weights to load.")
