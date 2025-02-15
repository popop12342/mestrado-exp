import csv
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader

LABELS = {
    'General Inquiry': 0,
    'Human Resources': 1,
    'Billing and Payments': 2,
    'Sales and Pre-Sales': 3,
    'IT Support': 4,
    'Customer Service': 5,
    'Product Support': 6,
    'Returns and Exchanges': 7,
    'Service Outages and Maintenance': 8,
    'Technical Support': 9
}


class HelpdeskDatasetLoader(AbstractDatasetLoader):
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        fraction = self._get_fraction(dataset_name)
        train_sentences, train_labels = HelpdeskDatasetLoader.load_from_file('../data/helpdesk/helpdesk-train.csv')
        test_sentences, test_labels = HelpdeskDatasetLoader.load_from_file('../data/helpdesk/helpdesk-test.csv')

        if fraction:
            train_sentences, train_labels = self.fraction_training_set(fraction, train_sentences, train_labels)

        return train_sentences, train_labels, test_sentences, test_labels

    @staticmethod
    def load_from_file(csv_file: str) -> tuple[list[str], list[str]]:
        sentences = []
        labels = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentences.append(row['text'])
                labels.append(LABELS[row['label']])
        return sentences, labels

    def get_labels(self) -> list[str]:
        return [str(val) for val in LABELS.values()]

    def get_label_names(self):
        return LABELS.keys()
