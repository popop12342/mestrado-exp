from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class LLMDatasetLoader(AbstractDatasetLoader):
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        base_dataset, fraction = self._get_base_dataset_and_fraction(dataset_name)
        dataset_folder = f'../data/llm{base_dataset}/'
        if not fraction:
            train_file = f'{dataset_folder}train_orig.txt'
        else:
            train_file = f'{dataset_folder}llm_{base_dataset}_{fraction}.txt'

        train_sentences, train_labels = self.load_from_file(train_file)
        test_sentences, test_labels = self.load_from_file(f'{dataset_folder}test.txt')

        return train_sentences, train_labels, test_sentences, test_labels

    def get_labels(self) -> list[str]:
        return ['0', '1']

    def load_from_file(self, filename: str) -> tuple[list[str], list[str]]:
        sentences = []
        labels = []
        with open(filename, 'r') as f:
            for line in f:
                splits = line.split('\t')
                label = splits[0]
                sentence = ' '.join(splits[1:])
                sentences.append(sentence)
                labels.append(label)
        return sentences, labels

    def _get_base_dataset_and_fraction(self, dataset_name: str) -> tuple[str, str]:
        splits = dataset_name.split('_')
        return splits[1], splits[2]
