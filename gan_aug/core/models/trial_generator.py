import torch
import torch.nn as nn

class TrialGenerator(nn.Module):
    def __init__(self, trial, noise_size=100, output_size=128):
        super(TrialGenerator, self).__init__()
        layers = []

        # study gantext
        self.num_layers = trial.suggest_int('generator_layers', 1, 8)
        self.hidden_size = trial.suggest_int('generator_hidden_size', 32, 256, 16)
        self.dropout = trial.suggest_float('generator_dropout', 0, 0.8)

        self.gru = nn.GRU(
            input_size=noise_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
            
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(2*self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def initHidden(self, batch_size, device):
        return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=device)
    
    def forward(self, noise, hidden):
        output, hidden = self.gru(noise, hidden)
        output = self.softmax(self.out(output))
        return output, hidden