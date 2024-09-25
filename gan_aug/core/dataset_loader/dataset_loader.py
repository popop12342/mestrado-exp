from typing import List, Tuple
from dataset_loader.subj_dataset_loader import SUBJDatasetLoader
from dataset_loader.aclimdb_dataset_loader import AclImdbDatasetLoader
from dataset_loader.rotten400k_dataset_loader import Rotten400kDatasetLoader
from dataset_loader.task_oriented_dialog_dataset_loader import TaskOrientedDialogDatasetLoader
from dataset_loader.olist_dataset_loader import OlistDatasetLoader
from dataset_loader.helpdesk_dataset_loader import HelpdeskDatasetLoader
from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader
from dataset_loader.turkish_product_reviews_dataset_loader import TurkishProductReviewsDatasetLoader


def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if dataset_name.startswith('subj'):
        return SUBJDatasetLoader.load(_get_fraction(dataset_name))
    elif dataset_name.startswith('aclImdb'):
        return AclImdbDatasetLoader.load(_get_fraction(dataset_name))
    elif dataset_name == 'rotten400k':
        return Rotten400kDatasetLoader.load()
    elif dataset_name.startswith('task-oriented-dialog'):
        lang, fraction = _get_lang_and_fraction(dataset_name)
        return TaskOrientedDialogDatasetLoader.load(lang=lang, fraction=fraction)
    elif dataset_name.startswith('olist'):
        return OlistDatasetLoader.load(_get_fraction(dataset_name))
    elif dataset_name.startswith('helpdesk'):
        return HelpdeskDatasetLoader.load(_get_fraction(dataset_name))
    elif dataset_name.startswith('turkish-product-reviews'):
        return TurkishProductReviewsDatasetLoader.load(_get_fraction(dataset_name))
    raise FileNotFoundError('Dataset with name {} was not found'.format(dataset_name))


def get_labels(dataset_name: str) -> list[str]:
    return _get_dataset_loader(dataset_name).get_labels()


_dataset_loaders: dict[str: AbstractDatasetLoader] = {
    'subj': SUBJDatasetLoader,
    'aclImdb': AclImdbDatasetLoader,
    'rotten400k': Rotten400kDatasetLoader,
    'task-oriented-dialog': TaskOrientedDialogDatasetLoader,
    'olist': OlistDatasetLoader,
    'helpdesk': HelpdeskDatasetLoader,
    'turkish-product-reviews': TurkishProductReviewsDatasetLoader
}


def _get_dataset_loader(dataset_name: str) -> AbstractDatasetLoader:
    name = dataset_name.split('_')[0]
    return _dataset_loaders[name]


def _get_fraction(dataset_name: str) -> str:
    fraction = None
    if '_' in dataset_name:
        fraction = '_'.join(dataset_name.split('_')[1:])
    return fraction


def _get_lang_and_fraction(dataset_name: str) -> tuple[str, float]:
    if '_' not in dataset_name:
        return None, None

    splits = dataset_name.split('_')
    lang = splits[1]
    fraction = int(splits[2]) / 100
    return lang, fraction
