from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class AbstractDatasetLoader(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def load() -> Tuple[List[str], List[str], List[str], List[str]]:
        """Loads the training set and validation set of one dataset

        Returns:
            List[str]: train sentences
            List[str]: train labels
            List[str]: test sentences
            List[str]: test labels
        """
        pass

    @staticmethod
    @abstractmethod
    def get_labels() -> list[str]:
        """Returns the labels of the dataset

        Returns:
            list[str]: list of labels
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
