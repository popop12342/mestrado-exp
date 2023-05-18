import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=128, hidden_sizes=[64], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        self.hidden_size = hidden_sizes[0]
        # hidden_sizes = [noise_size] + hidden_sizes

        self.gru = nn.GRU(input_size=noise_size, hidden_size=self.hidden_size, batch_first=True)

        # for i in range(len(hidden_sizes) - 1):
        #     layers.extend([
        #         nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), 
        #         nn.LeakyReLU(0.2, inplace=True), 
        #         nn.Dropout(dropout_rate)
        #         ])
            
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
    def forward(self, noise, hidden):
        output, hidden = self.gru(noise, hidden)
        # output = self.softmax(self.out(output))
        output = self.out(output)
        return output, hidden