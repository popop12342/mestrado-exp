import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, trail, noise_size=100, output_size=128):
        super(Generator, self).__init__()
        layers = []

        # study gantext
        # self.num_layers = trail.suggest_int('generator_layers', 1, 4)
        # self.hidden_size = trail.suggest_int('generator_hidden_size', 32, 128, 16)
        self.num_layers = 1#4
        self.hidden_size = 96

        self.gru = nn.GRU(
            input_size=noise_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
            
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def forward(self, noise, hidden):
        output, hidden = self.gru(noise, hidden)
        output = self.softmax(self.out(output))
        return output, hidden