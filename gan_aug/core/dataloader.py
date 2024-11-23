import torch

from typing import List, Tuple
import numpy as np
from transformers import PreTrainedTokenizer

from dataset_loader.dataset_loader import load_dataset
from eda import eda


MAX_SEQ_SIZE = 64


def augment(sentences: List[str], labels: List[str], num_aug: int) -> Tuple[List[str], List[str]]:
    augmented_sentences = []
    augmented_labels = []
    for sentence, label in zip(sentences, labels):
        augmented_sentences.extend(eda(sentence, num_aug=num_aug))
        augmented_labels.extend([label] * (num_aug + 1))
    print('Augmenting original dataset by {}, from {} to {}'.format(num_aug+1, len(sentences), len(augmented_sentences)))
    return augmented_sentences, augmented_labels


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
