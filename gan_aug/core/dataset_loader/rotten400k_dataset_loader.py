import csv

from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class Rotten400kDatasetLoader(AbstractDatasetLoader):

    POSITIVE_TRESHOLD = 50
    TRAIN_SPLIT = 0.9

    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)
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

        if fraction:
            train_sentences, train_labels = self.fraction_training_set(fraction, train_sentences, train_labels)

        return train_sentences, train_labels, test_sentences, test_labels

    def get_labels(self) -> list[str]:
        return ['0', '1']
