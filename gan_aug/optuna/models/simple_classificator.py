import torch.nn as nn

class SimpleClassificator(nn.Module):
    def __init__(self, num_layers, hidden=64, dropout_rate=0.1, linear_size=32, input_size=128, num_labels=2):
        super(SimpleClassificator, self).__init__()
        
        self.layers = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        self.linear = nn.Linear(2*hidden, linear_size)
        self.relu = nn.ReLU()
        self.logit = nn.Linear(linear_size, num_labels + 1) # + 1 for label being fake/real
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, text):
        x, _ = self.layers(text.float())
        x = self.linear(x[:,-1,:])
        x = self.relu(x)
        logits = self.logit(x)
        probs = self.softmax(logits)
        return x, logits, probs