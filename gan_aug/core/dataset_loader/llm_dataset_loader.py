from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader


class LLMDatasetLoader(AbstractDatasetLoader):
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        base_dataset, fraction = self._get_base_dataset_and_fraction(dataset_name)
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []
        dataset_folder = f'../data/llm{base_dataset}/'
        if not fraction:
            train_file = f'{dataset_folder}train_orig.txt'
        else:
            train_file = f'{dataset_folder}llm_{base_dataset}_{fraction}.txt'
        with open(train_file, 'r') as f:
            for line in f:
                splits = line.split('\t')
                label = splits[0]
                sentence = ' '.join(splits[1:])
                train_sentences.append(sentence)
                train_labels.append(label)
        with open(f'{dataset_folder}test.txt', 'r') as f:
            for line in f:
                splits = line.split('\t')
                label = splits[0]
                sentence = ' '.join(splits[1:])
                test_sentences.append(sentence)
                test_labels.append(label)
        return train_sentences, train_labels, test_sentences, test_labels

    def get_labels(self) -> list[str]:
        return ['0', '1']

    def _get_base_dataset_and_fraction(self, dataset_name: str) -> tuple[str, str]:
        splits = dataset_name.split('_')
        return splits[1], splits[2]
