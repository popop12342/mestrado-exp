import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, vocab_size, padding_idx, input_size=128, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, input_size, padding_idx=padding_idx)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1], num_labels + 1) # + 1 for label being fake/real
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, text):
        embedded = self.embedding(text)
        last_rep = self.layers(embedded)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs