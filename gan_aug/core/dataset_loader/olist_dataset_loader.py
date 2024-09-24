from typing import List, Tuple
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class OlistDatasetLoader(AbstractDatasetLoader):
    @staticmethod
    def load(fraction: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []

        with open('../data/olist/olist-train.tsv', 'r') as f:
            for line in f:
                label, sentence = line.split('\t')
                train_sentences.append(sentence)
                train_labels.append(label)
        with open('../data/olist/olist-test.tsv', 'r') as f:
            for line in f:
                label, sentence = line.split('\t')
                test_sentences.append(sentence)
                test_labels.append(label)

        if fraction:
            fraction = int(fraction) / 100
            num_train = int(fraction * len(train_sentences))
            train_sentences = train_sentences[:num_train]
            train_labels = train_labels[:num_train]

        return train_sentences, train_labels, test_sentences, test_labels
