import csv

from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader
from typing import List, Tuple


class Rotten400kDatasetLoader(AbstractDatasetLoader):

    POSITIVE_TRESHOLD = 50
    TRAIN_SPLIT = 0.9

    @staticmethod
    def load() -> Tuple[List[str], List[str], List[str], List[str]]:
        sentences = []
        labels = []
        with open('../data/rotten400k/rottentomatoes-400k.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                sentences.append(row['Review'])
                label = '1' if float(row['Score']) > Rotten400kDatasetLoader.POSITIVE_TRESHOLD else '0'
                labels.append(label)

        train_size = int(len(sentences) * Rotten400kDatasetLoader.TRAIN_SPLIT)
        train_sentences = sentences[:train_size]
        train_labels = labels[:train_size]
        test_sentences = sentences[train_size:]
        test_labels = labels[train_size:]

        return train_sentences, train_labels, test_sentences, test_labels

    @staticmethod
    def get_labels() -> List[str]:
        return ['0', '1']
