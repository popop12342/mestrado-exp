import torch
from argparse import ArgumentParser

from dataloader import create_dataloaders

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def generate(file: str, dataset: str, num: int):
    _, _, seq_size, vocab = create_dataloaders(dataset, device=device)
    generator = torch.load(file, map_location=device)

    for _ in range(num):
        noise = torch.zeros(1, seq_size, len(vocab), device=device).uniform_(0, 1)
        hidden = generator.initHidden(1, device)
        gen_out, hidden = generator(noise, hidden)
        gen_rep = torch.argmax(gen_out, dim=2)
        tokens = vocab.lookup_tokens(gen_rep[0,:].numpy())
        print(' '.join(tokens))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--dataset')
    parser.add_argument('-n', help='Number of samples to produce', default=1, type=int)
    args = parser.parse_args()

    generate(args.model, args.dataset, args.n)

