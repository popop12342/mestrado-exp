from dataset_loader.abstract_dataset_loader import AbstractDatasetLoader
from typing import Tuple, List

import os

DATASET_BASE = '../data/aclImdb'

class AclImdbDatasetLoader(AbstractDatasetLoader):

    @staticmethod
    def load(fraction: str = None) -> Tuple[List[str], List[str], List[str], List[str]]:
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []
        
        train_dir = os.path.join(DATASET_BASE, 'train')
        # positive train sentences
        pos_train_dir = os.path.join(train_dir, 'pos')
        train_sentences.extend(AclImdbDatasetLoader._load_files_from(pos_train_dir, fraction))
        train_labels += ['1'] * len(train_sentences)

        # negative train sentences
        neg_train_dir = os.path.join(train_dir, 'neg')
        train_sentences.extend(AclImdbDatasetLoader._load_files_from(neg_train_dir, fraction))
        train_labels += ['0'] * (len(train_sentences) - len(train_labels))

        test_dir = os.path.join(DATASET_BASE, 'test')
        #positive test sentences
        pos_test_dir = os.path.join(test_dir, 'pos')
        test_sentences.extend(AclImdbDatasetLoader._load_files_from(pos_test_dir))
        test_labels += ['1'] * len(test_sentences)

        # negative test sentences
        neg_test_dir = os.path.join(test_dir, 'neg')
        test_sentences.extend(AclImdbDatasetLoader._load_files_from(neg_test_dir))
        test_labels += ['0'] * (len(test_sentences) - len(test_labels))

        return train_sentences, train_labels, test_sentences, test_labels


    @staticmethod 
    def _load_files_from(dirpath: str, fraction: str = None) -> List[str]:
        files = os.listdir(dirpath)
        if fraction:
            keep_percent = int(fraction) / 100
            num_files = int(keep_percent * len(files))
            files = files[:num_files]
        content = []
        for filename in files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r') as f:
                content.append(f.readline())
        return content
