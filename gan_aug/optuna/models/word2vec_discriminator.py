import torch
import numpy as np
import torch.nn as nn

class Word2VecDiscriminator(nn.Module):
    def __init__(self, trial, word2vec, vocab, input_size=128, num_labels=2):
        super(Word2VecDiscriminator, self).__init__()
        self.word2vec = word2vec
        self.word2vec_size = 300
        self.vocab = vocab
        
        num_layers = trial.study.user_attrs['num_layers']
        hidden = 64
        dropout_rate = 0.1
        linear_size = 32

        self.layers = nn.LSTM(
            input_size=self.word2vec_size,
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
        batch_size, seq_size = text.size()
        X = np.zeros((batch_size, seq_size, self.word2vec_size))
        for i, sentence in enumerate(text):
            offset = seq_size - len(sentence)
            for j, idx in enumerate(sentence):
                token = self.vocab.lookup_token(idx)
                if token in self.word2vec:
                    X[i, offset+j, :] = self.word2vec[token]
        X = torch.tensor(X)

        x, _ = self.layers(X.float())
        x = self.linear(x[:,-1,:])
        x = self.relu(x)
        logits = self.logit(x)
        probs = self.softmax(logits)
        return x, logits, probs