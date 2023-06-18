from typing import List, Tuple
from dataset_loader.subj_dataset_loader import SUBJDatasetLoader
from dataset_loader.aclimdb_dataset_loader import AclImdbDatasetLoader
from dataset_loader.rotten400k_dataset_loader import Rotten400kDatasetLoader

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if dataset_name.startswith('subj'):
        fraction = None
        if '_' in dataset_name:
            fraction = '_'.join(dataset_name.split('_')[1:])
        return SUBJDatasetLoader.load(fraction)
    elif dataset_name == 'aclImdb':
        return AclImdbDatasetLoader.load()
    elif dataset_name == 'rotten400k':
        return Rotten400kDatasetLoader.load()
    raise FileNotFoundError('Dataset with name {} was not found'.format(dataset_name))