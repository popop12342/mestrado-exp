from .llm_dataset_loader import LLMDatasetLoader


class ConfigurableLLMDatasetLoader(LLMDatasetLoader):

    def __init__(self, base_dataset: str, subdir: str):
        self.base_dataset = base_dataset
        self.subdir = subdir

    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)

        dataset_folder = f'../data/llm{self.base_dataset}/{self.subdir}/'
        train_file = f'{dataset_folder}llm_{self.base_dataset}_{fraction}.txt'

        train_sentences, train_labels = self.load_from_file(train_file)
        test_sentences, test_labels = self.load_from_file(f'{dataset_folder}test.txt')

        return train_sentences, train_labels, test_sentences, test_labels
