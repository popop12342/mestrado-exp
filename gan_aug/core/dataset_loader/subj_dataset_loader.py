from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class SUBJDatasetLoader(AbstractDatasetLoader):
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []
        if not fraction:
            train_file = '../data/subj/train_orig.txt'
        else:
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

    def get_labels(self) -> list[str]:
        return ['0', '1']

    def get_label_names(self):
        return ['subjective', 'objective']
