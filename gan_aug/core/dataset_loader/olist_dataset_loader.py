from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class OlistDatasetLoader(AbstractDatasetLoader):
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)
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
            train_sentences, train_labels = self.fraction_training_set(fraction, train_sentences, train_labels)

        return train_sentences, train_labels, test_sentences, test_labels

    def get_labels(self) -> list[str]:
        return ['0', '1']

    def get_label_names(self):
        return ['negative', 'positive']
