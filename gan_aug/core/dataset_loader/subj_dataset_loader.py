from typing import List, Tuple
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader

class SUBJDatasetLoader(AbstractDatasetLoader):
    @staticmethod
    def load(fraction: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []
        if not fraction:
            train_file = '../data/subj/train_orig.txt'
        else :
            train_file = '../data/subj/train_{}.txt'.format(fraction)
        with open(train_file, 'r') as f:
            for line in f:
                label, sentence = line.split('\t')
                train_sentences.append(sentence)
                train_labels.append(label)
        with open('../data/subj/test.txt', 'r') as f:
            for line in f:
                label, sentence = line.split('\t')
                test_sentences.append(sentence)
                test_labels.append(label)
        return train_sentences, train_labels, test_sentences, test_labels