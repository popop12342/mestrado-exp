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
