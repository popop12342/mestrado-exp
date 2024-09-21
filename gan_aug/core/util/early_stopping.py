class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                print(f"Initial score set at {current_score:.6f}")
        elif current_score < self.best_score + self.min_delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
        else:
            self.best_score = current_score
            self.epochs_no_improve = 0  # Reset counter if there is improvement
            if self.verbose:
                print(f"Improvement found: {current_score:.6f} (previous best: {self.best_score:.6f})")
