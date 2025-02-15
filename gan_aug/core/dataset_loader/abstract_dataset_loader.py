from abc import ABCMeta, abstractmethod


class AbstractDatasetLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self, dataset_name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        """Loads the training set and validation set of one dataset

        Args:
            dataset_name (str): dataset name

        Returns:
            list[str]: train sentences
            list[str]: train labels
            list[str]: test sentences
            list[str]: test labels
        """
        pass

    @abstractmethod
    def get_labels(self) -> list[str]:
        """Returns the labels of the dataset

        Returns:
            list[str]: list of labels
        """
        pass

    @abstractmethod
    def get_label_names(self) -> list[str]:
        """Returns the labels names of the dataset, on the same order as `get_labels` method.

        Returns:
            list[str]: list of label names
        """
        pass

    @staticmethod
    def fraction_training_set(fraction: str, train_sentences: list[str],
                              train_labels: list[str]) -> tuple[list[str], list[str]]:
        """Fraction the training set to reduce the number of available samples, for simulating low resource scenarios

        Args:
            fraction (str): fraction amount to keep of trainin dataset, eg "001" for 1%
            train_sentences (list[str]): training set sentences
            train_labels (list[str]): training set labels

        Returns:
            tuple[list[str], list[str]]: sentences and labels fractioned
        """
        fraction = int(fraction) / 100
        num_train = int(fraction * len(train_sentences))
        train_sentences = train_sentences[:num_train]
        train_labels = train_labels[:num_train]
        return train_sentences, train_labels

    def _get_fraction(self, dataset_name: str) -> str:
        fraction = None
        if '_' in dataset_name:
            fraction = '_'.join(dataset_name.split('_')[1:])
        return fraction

    def _get_lang_and_fraction(self, dataset_name: str) -> tuple[str, float]:
        if '_' not in dataset_name:
            return None, None

        splits = dataset_name.split('_')
        lang = splits[1]
        fraction = int(splits[2]) / 100
        return lang, fraction
