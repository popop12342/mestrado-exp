import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, trial, vocab_size, padding_idx, input_size=128, num_labels=2):
        super(Discriminator, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, input_size, padding_idx=padding_idx)
        layers = []
        
        # study gantext
        # num_layers = trail.suggest_int('discriminator_layers', 1, 4)
        # hidden = trail.suggest_int('discriminator_hidden_size', 32, 128, 16)
        num_layers = 1#trial.study.user_attrs['num_layers']
        hidden = 64
        dropout_rate = 0.1
        linear_size = 32
        
        # hidden_sizes = [input_size] + [hidden] * num_layers
        # for i in range(len(hidden_sizes) - 1):
        #     layers.extend([
        #         nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
        #         nn.LeakyReLU(dropout_rate, inplace=True)
        #     ])
        # self.layers = nn.Sequential(*layers)
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
        # embedded = self.embedding(text)
        # last_rep = self.layers(embedded)
        # logits = self.logit(last_rep)
        x, _ = self.layers(text.float())
        x = self.linear(x[:,-1,:])
        x = self.relu(x)
        logits = self.logit(x)
        probs = self.softmax(logits)
        return x, logits, probs
        # return last_rep, logits, probs