from typing import List, Tuple
from dataset_loader.subj_dataset_loader import SUBJDatasetLoader
from dataset_loader.aclimdb_dataset_loader import AclImdbDatasetLoader

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if dataset_name == 'subj':
        return SUBJDatasetLoader.load()
    elif dataset_name == 'aclImdb':
        return AclImdbDatasetLoader.load()
    raise FileNotFoundError('Dataset with name {} was not found'.format(dataset_name))