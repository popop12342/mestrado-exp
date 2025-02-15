import os
import random
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


BASE_DIR = '../data/task_oriented_dialog'
LABELS = {
    'alarm': 0,
    'reminder': 1,
    'weather': 2
}


class TaskOrientedDialogDatasetLoader(AbstractDatasetLoader):

    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        lang, fraction = self._get_lang_and_fraction(dataset_name)
        if lang not in ['en', 'es', 'th']:
            raise FileNotFoundError(f'Task oriented dialog dataset has no language {lang}')

        lang_dir = os.path.join(BASE_DIR, lang)
        if lang == 'th':
            train_file = os.path.join(lang_dir, 'train-th_TH.tsv')
            test_file = os.path.join(lang_dir, 'test-th_TH.tsv')
        else:
            train_file = os.path.join(lang_dir, f'train-{lang}.tsv')
            test_file = os.path.join(lang_dir, f'test-{lang}.tsv')

        train_sentences, train_labels = TaskOrientedDialogDatasetLoader._load_from_file(train_file)
        test_sentences, test_labels = TaskOrientedDialogDatasetLoader._load_from_file(test_file)

        if fraction:
            train_data = list(zip(train_sentences, train_labels))
            random.shuffle(train_data)
            num_train = int(fraction * len(train_sentences))
            train_data = train_data[:num_train]
            train_sentences = [x[0] for x in train_data]
            train_labels = [x[1] for x in train_data]

        return train_sentences, train_labels, test_sentences, test_labels

    @staticmethod
    def _load_from_file(file: str) -> tuple[list[str], list[str]]:
        sentences = []
        labels = []

        with open(file, 'r') as f:
            for line in f:
                cols = line.split('\t')
                domain, intent = cols[0].split('/')
                text = cols[2]
                label = LABELS[domain]
                sentences.append(text)
                labels.append(label)

        return sentences, labels

    def get_labels(self) -> list[str]:
        return ['0', '1', '2']

    def get_label_names(self):
        return ['alarm', 'reminder', 'weather']
