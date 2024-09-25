from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class OlistDatasetLoader(AbstractDatasetLoader):
    @staticmethod
    def load(fraction: str) -> tuple[list[str], list[str], list[str], list[str]]:
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
            train_sentences, train_labels = AbstractDatasetLoader.fraction_training_set(fraction, train_sentences,
                                                                                        train_labels)

        return train_sentences, train_labels, test_sentences, test_labels

    @staticmethod
    def get_labels() -> list[str]:
        return ['0', '1']
