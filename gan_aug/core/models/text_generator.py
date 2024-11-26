import torch
import torch.nn as nn


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, max_seq_len, device):
        super(TextGenerator, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, trg):
        """
        Parameters:
        - src: Tensor (batch_size, seq_len), input tokens
        - trg: Tensor (batch_size, seq_len), target tokens (teacher-forced during training)

        Returns:
        - output: Tensor (batch_size, seq_len, vocab_size), logits for each token
        """
        # Embedding
        src_emb = self.embedding(src) * (self.max_seq_len ** 0.5)  # scale embeddings
        trg_emb = self.embedding(trg) * (self.max_seq_len ** 0.5)

        # Add positional encodings
        src_emb = self.add_positional_encoding(src_emb)
        trg_emb = self.add_positional_encoding(trg_emb)

        # Transformer Encoding
        memory = self.encoder(src_emb.transpose(0, 1))

        # Transformer Decoding
        output = self.decoder(trg_emb.transpose(0, 1), memory)

        # Project to vocab space
        logits = self.fc_out(output.transpose(0, 1))

        return logits

    def add_positional_encoding(self, x):
        """
        Adds positional encoding to embeddings.
        """
        batch_size, seq_len, dim = x.size()
        position = torch.arange(0, seq_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=self.device) * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_enc = torch.zeros((seq_len, dim), device=self.device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return x + pos_enc.unsqueeze(0)

    def generate(self, src, max_len):
        """
        Generates text given a source input.
        Parameters:
        - src: Tensor (1, seq_len), input tokens
        - max_len: int, maximum length to generate
        Returns:
        - generated_seq: Tensor, generated token indices
        """
        src_emb = self.embedding(src) * (self.max_seq_len ** 0.5)
        src_emb = self.add_positional_encoding(src_emb)
        memory = self.encoder(src_emb.transpose(0, 1))

        generated_seq = [torch.tensor([2], device=self.device)]  # Start token
        for _ in range(max_len):
            trg = torch.cat(generated_seq, dim=0).unsqueeze(0)
            trg_emb = self.embedding(trg) * (self.max_seq_len ** 0.5)
            trg_emb = self.add_positional_encoding(trg_emb)
            output = self.decoder(trg_emb.transpose(0, 1), memory)
            logits = self.fc_out(output.transpose(0, 1))
            next_token = logits[:, -1, :].argmax(dim=1, keepdim=True)
            if next_token.item() == 3:  # End token
                break
            generated_seq.append(next_token)
        return torch.cat(generated_seq, dim=1).squeeze(0)
