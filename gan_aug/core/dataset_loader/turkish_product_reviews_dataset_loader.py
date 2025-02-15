import random
from datasets import load_dataset
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader

TRAIN_SPLIT = 0.9


class TurkishProductReviewsDatasetLoader(AbstractDatasetLoader):

    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)
        dataset = load_dataset('fthbrmnby/turkish_product_reviews')
        sentences = dataset['train']['sentence']
        labels = dataset['train']['sentiment']
        dataset = list(zip(sentences, labels))
        random.shuffle(dataset)
        sentences = [data[0] for data in dataset]
        labels = [str(data[1]) for data in dataset]

        num_train = int(TRAIN_SPLIT * len(sentences))
        train_sentences = sentences[:num_train]
        train_labels = labels[:num_train]
        test_sentences = sentences[num_train:]
        test_labels = labels[num_train:]

        if fraction:
            train_sentences, train_labels = self.fraction_training_set(fraction, train_sentences, train_labels)

        return train_sentences, train_labels, test_sentences, test_labels

    def get_labels(self) -> list[str]:
        return ['0', '1']

    def get_label_names(self):
        return ['negative', 'positive']
