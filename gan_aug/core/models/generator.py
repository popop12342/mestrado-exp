import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_layers, noise_size=100, output_size=128, hidden_size=64, dropout=0.1,
                 initial_temperature=1.0):
        super(Generator, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.temperature = initial_temperature

        self.gru = nn.GRU(
            input_size=noise_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, noise, hidden):
        output, hidden = self.gru(noise, hidden)
        logits = self.out(output)
        gumbel_out = nn.functional.gumbel_softmax(logits, self.temperature)
        return gumbel_out

    def set_temperature(self, temperature):
        self.temperature = temperature
