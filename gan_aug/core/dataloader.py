import pickle
import torch
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from argparse import ArgumentParser
from typing import List, Tuple
import numpy as np
from transformers import PreTrainedTokenizer

from dataset_loader.dataset_loader import load_dataset
from eda import eda

# last version of torchtext warning
torchtext.disable_torchtext_deprecation_warning()

tokenizer = get_tokenizer('basic_english')
MAX_SEQ_SIZE = 64


def build_vocab(train_sentences):
    def yield_tokens(sentences):
        for sen in sentences:
            yield tokenizer(sen)

    vocab = build_vocab_from_iterator(yield_tokens(train_sentences), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def encode_xy(sentences, labels, vocab, seq_size, device):
    input_ids = []
    label_ids = []
    label_mask = []

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) if x != 'UNK' else 0

    for (text, label) in zip(sentences, labels):
        label_ids.append(label_pipeline(label))
        label_mask.append(label != 'UNK')
        processed_text = text_pipeline(text)
        if (seq_size > len(processed_text)):
            processed_text = [vocab['<pad>']] * (seq_size - len(processed_text)) + processed_text
        else:
            processed_text = processed_text[:seq_size]
        input_ids.append(processed_text)
    x = torch.tensor(input_ids, dtype=torch.int32, device=device)
    y = torch.tensor(label_ids, device=device)
    mask_y = torch.tensor(label_mask, device=device)
    return x, y, mask_y


def create_dataset(sentences, labels, vocab, seq_size, device):
    input_ids, label_ids, label_mask = encode_xy(sentences, labels, vocab, seq_size, device)
    return torch.utils.data.TensorDataset(input_ids, label_ids, label_mask)


def create_dataloaders(dataset_name: str, batch_size: int = 8, device: str = 'cpu', num_aug: int = 0):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset_name)
    print('Using dataset ' + dataset_name)

    if num_aug > 0:
        train_sentences, train_labels = augment(train_sentences, train_labels, num_aug)

    vocab = build_vocab(train_sentences)
    seq_size = max([len(tokenizer(text)) for text in train_sentences])
    seq_size = min(seq_size, MAX_SEQ_SIZE)
    print('Seq size is ' + str(seq_size) + ' maximum is ' + str(MAX_SEQ_SIZE))

    train_dataset = create_dataset(train_sentences, train_labels, vocab, seq_size, device)
    test_dataset = create_dataset(test_sentences, test_labels, vocab, seq_size, device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader, seq_size, vocab


def augment(sentences: List[str], labels: List[str], num_aug: int) -> Tuple[List[str], List[str]]:
    augmented_sentences = []
    augmented_labels = []
    for sentence, label in zip(sentences, labels):
        augmented_sentences.extend(eda(sentence, num_aug=num_aug))
        augmented_labels.extend([label] * (num_aug + 1))
    print('Augmenting original dataset by {}, from {} to {}'.format(num_aug+1, len(sentences), len(augmented_sentences)))
    return augmented_sentences, augmented_labels


def create_word2vec_dataloaders(dataset_name: str, batch_size: int = 8, device: str = 'cpu', num_aug: int = 0):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset_name)
    print('Using dataset ' + dataset_name)

    if num_aug > 0:
        train_sentences, train_labels = augment(train_sentences, train_labels, num_aug)

    vocab = build_vocab(train_sentences + test_sentences)
    seq_size = max([len(tokenizer(text)) for text in train_sentences])
    seq_size = min(seq_size, MAX_SEQ_SIZE)
    print('Seq size is ' + str(seq_size) + ' maximum is ' + str(MAX_SEQ_SIZE))

    word2vec = load_word2vec(vocab)

    train_dataset = create_word2vec_dataset(train_sentences, train_labels, seq_size, device, word2vec, 300)
    test_dataset = create_word2vec_dataset(test_sentences, test_labels, seq_size, device, word2vec, 300)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader, seq_size, vocab


def create_bert_dataloaders(dataset_name: str, tokenizer: PreTrainedTokenizer, batch_size: int = 8, device: str = 'cpu', num_aug: int = 0):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset_name)
    print('Using dataset ' + dataset_name)

    if num_aug > 0:
        train_sentences, train_labels = augment(train_sentences, train_labels, num_aug)

    train_dataset = create_bert_dataset(train_sentences, train_labels, MAX_SEQ_SIZE, device, tokenizer)
    test_dataset = create_bert_dataset(test_sentences, test_labels, MAX_SEQ_SIZE, device, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader, MAX_SEQ_SIZE


def load_word2vec(vocab):
    text_embeddings = open('../word2vec/glove.840B.300d.txt', 'r').readlines()
    word2vec = {}
    for line in text_embeddings:
        items = line.split(' ')
        word = items[0]
        if word in vocab:
            word2vec[word] = np.asarray(items[1:], dtype='float32')
    return word2vec


def create_word2vec_dataset(sentences, labels, seq_size, device, word2vec, word2vec_len):
    X = np.zeros((len(sentences), seq_size, word2vec_len))
    label_ids = []

    label_pipeline = lambda x: int(x)

    for i in range(len(sentences)):
        text = sentences[i]
        label = labels[i]
        label_ids.append(label_pipeline(label))
        processed_text = tokenizer(text)[:seq_size]
        offset = seq_size - len(processed_text)
        for j, word in enumerate(processed_text):
            if word in word2vec:
                X[i, offset+j, :] = word2vec[word]

    X = torch.as_tensor(X, device=device)
    label_ids = torch.tensor(label_ids, device=device)
    return torch.utils.data.TensorDataset(X, label_ids)


def create_bert_dataset(sentences, labels, seq_size, device, tokenizer):
    label_ids = []
    label_mask = []
    input_ids = []
    input_mask_array = []
    label_pipeline = lambda x: int(x) if x != 'UNK' else 0

    for text, label in zip(sentences, labels):
        label_ids.append(label_pipeline(label))
        label_mask.append(label != 'UNK')
        encoded_sent = tokenizer.encode(text, add_special_tokens=True, max_length=seq_size, padding="max_length", truncation=True)
        input_ids.append(encoded_sent)
    
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_mask_array.append(att_mask)

    input_ids = torch.tensor(input_ids, device=device)
    input_mask_array = torch.tensor(input_mask_array, device=device)
    label_ids = torch.tensor(label_ids, device=device)
    mask_y = torch.tensor(label_mask, device=device)
    return torch.utils.data.TensorDataset(input_ids, input_mask_array, label_ids, mask_y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='subj')
    parser.add_argument('--pickle_file', default='../data/subj.pkl')
    args = parser.parse_args()

    dataloaders = create_word2vec_dataloaders(args.dataset)
    with open(args.pickle_file, 'wb') as pickle_file:
        pickle.dump(dataloaders, pickle_file)