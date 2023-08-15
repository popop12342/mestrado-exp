import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class BERTDiscriminator(nn.Module):
    def __init__(self, num_layers, seq_size, device, num_labels=2, dropout=0.1):
        super(BERTDiscriminator, self).__init__()
        model_name = "bert-base-cased"
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = int(AutoConfig.from_pretrained(model_name).hidden_size)

        self.seq_size = seq_size
        self.device = device
        self.hidden = hidden_size
        self.dropout = dropout
        self.linear_size = hidden_size
        self.count = 0

        self.input_dropout = nn.Dropout(p=self.dropout)
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Linear(self.hidden, self.hidden), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(self.dropout)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(self.hidden,num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, input_mask_array):
        self.count += 1
        
        model_outputs = self.transformer(input_ids, attention_mask=input_mask_array)
        hidden_states = model_outputs[-1]
    
        input_rep = self.input_dropout(hidden_states)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs