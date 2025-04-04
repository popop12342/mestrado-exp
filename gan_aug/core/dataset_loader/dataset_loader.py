from typing import List, Tuple

from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader
from dataset_loader.aclimdb_dataset_loader import AclImdbDatasetLoader
from dataset_loader.helpdesk_dataset_loader import HelpdeskDatasetLoader
from dataset_loader.multilingual_sentiments_dataset_loader import MultilingualSentimentsDatasetLoader
from dataset_loader.olist_dataset_loader import OlistDatasetLoader
from dataset_loader.rotten400k_dataset_loader import Rotten400kDatasetLoader
from dataset_loader.subj_dataset_loader import SUBJDatasetLoader
from dataset_loader.task_oriented_dialog_dataset_loader import \
    TaskOrientedDialogDatasetLoader
from dataset_loader.turkish_product_reviews_dataset_loader import \
    TurkishProductReviewsDatasetLoader
from dataset_loader.llm_dataset_loader import LLMDatasetLoader
from dataset_loader.configurable_llm_dataset_loader import ConfigurableLLMDatasetLoader

_dataset_loaders: dict[str: AbstractDatasetLoader] = {
    'subj': SUBJDatasetLoader(),
    'llm': LLMDatasetLoader(),
    'aclImdb': AclImdbDatasetLoader(),
    'rotten400k': Rotten400kDatasetLoader(),
    'task-oriented-dialog': TaskOrientedDialogDatasetLoader(),
    'olist': OlistDatasetLoader(),
    'helpdesk': HelpdeskDatasetLoader(),
    'turkish-product-reviews': TurkishProductReviewsDatasetLoader(),
    'multilingual-sentiments': MultilingualSentimentsDatasetLoader(),
    'cllmolist': ConfigurableLLMDatasetLoader('olist', 'samples-20_naug-1'),
    'cllmsubj': ConfigurableLLMDatasetLoader('subj', 'samples-5_naug-5'),
    'cllmaclImdb': ConfigurableLLMDatasetLoader('aclImdb', 'samples-5_naug-4'),
    'cllmrotten400k': ConfigurableLLMDatasetLoader('rotten400k', 'samples-10_naug-1'),
    'cllmhelpdesk': ConfigurableLLMDatasetLoader('helpdesk', 'samples-5_naug-1'),
}


def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    loader = _get_dataset_loader(dataset_name)
    if loader:
        return loader.load(dataset_name)
    raise FileNotFoundError('Dataset with name {} was not found'.format(dataset_name))


def get_labels(dataset_name: str) -> list[str]:
    dataset_loader = _get_dataset_loader(dataset_name)
    if isinstance(dataset_loader, ConfigurableLLMDatasetLoader):
        base_dataset_name = dataset_loader.base_dataset
        return _get_dataset_loader(base_dataset_name).get_labels()
    return dataset_loader.get_labels()


def get_label_names(dataset_name: str) -> list[str]:
    dataset_loader = _get_dataset_loader(dataset_name)
    if isinstance(dataset_loader, ConfigurableLLMDatasetLoader):
        base_dataset_name = dataset_loader.base_dataset
        return _get_dataset_loader(base_dataset_name).get_label_names()
    return dataset_loader.get_label_names()


def _get_dataset_loader(dataset_name: str) -> AbstractDatasetLoader:
    name = dataset_name.split('_')[0]
    return _dataset_loaders[name]
