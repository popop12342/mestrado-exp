from datasets import load_dataset
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class MultilingualSentimentsDatasetLoader(AbstractDatasetLoader):

    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        lang, fraction = self._get_lang_and_fraction(dataset_name)
        dataset = load_dataset('tyqiangz/multilingual-sentiments', lang)
        train_sentences = dataset['train']['text']
        train_labels = dataset['train']['label']

        print('Training size: ' + str(len(train_sentences)))

        if fraction:
            train_sentences, train_labels = self.fraction_training_set(fraction * 100, train_sentences, train_labels)

        print('Training size: ' + str(len(train_sentences)))

        return train_sentences, train_labels, dataset['test']['text'], dataset['test']['label']

    def get_labels(self) -> list[str]:
        return ['0', '1', '2']

    def get_label_names(self):
        return ['negative', 'neutral', 'positive']
