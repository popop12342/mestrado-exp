import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_layers, noise_size=100, output_size=128, hidden_size=64, dropout=0.1):
        super(Generator, self).__init__()
        layers = []

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

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