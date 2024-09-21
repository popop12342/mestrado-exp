import os
import random

from create_datasets import get_only_chars

DATASET_BASE = '../gan_aug/data/aclImdb'

def _load_files_from(dirpath: str, fraction: str = None):
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


def load(fraction: str = None):
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    
    train_dir = os.path.join(DATASET_BASE, 'train')
    # positive train sentences
    pos_train_dir = os.path.join(train_dir, 'pos')
    train_sentences.extend(_load_files_from(pos_train_dir, fraction))
    train_labels += ['1'] * len(train_sentences)

    # negative train sentences
    neg_train_dir = os.path.join(train_dir, 'neg')
    train_sentences.extend(_load_files_from(neg_train_dir, fraction))
    train_labels += ['0'] * (len(train_sentences) - len(train_labels))

    test_dir = os.path.join(DATASET_BASE, 'test')
    #positive test sentences
    pos_test_dir = os.path.join(test_dir, 'pos')
    test_sentences.extend(_load_files_from(pos_test_dir))
    test_labels += ['1'] * len(test_sentences)

    # negative test sentences
    neg_test_dir = os.path.join(test_dir, 'neg')
    test_sentences.extend(_load_files_from(neg_test_dir))
    test_labels += ['0'] * (len(test_sentences) - len(test_labels))

    return train_sentences, train_labels, test_sentences, test_labels

def create_lines(sentences, labels):
    lines = []
    for sent, label in zip(sentences, labels):
        sent = get_only_chars(sent)
        lines.append('ACLIMDB:' + label + ' ' + sent + '\n')
    return lines


def create_dataset_folder(increment):
    train_sentences, train_labels, test_sentences, test_labels = load()

    train_lines = create_lines(train_sentences, train_labels)
    test_lines = create_lines(test_sentences, test_labels)

    dir_name = 'data/aclImdb/aclImdb_{:03d}'.format(int(100 * increment))
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    train_size = int(len(train_lines) * increment)
    random.shuffle(train_lines)
    train_lines = train_lines[:train_size]

    with open(os.path.join(dir_name, 'labeled.tsv'), 'w') as f:
        f.write('fine_label utterance\n')
        f.writelines(train_lines)

    with open(os.path.join(dir_name, 'test.tsv'), 'w') as f:
        f.write('fine_label utterance\n')
        f.writelines(test_lines)
    
    with open(os.path.join(dir_name, 'unlabeled.tsv'), 'w') as f:
        f.write('fine_label utterance\n')

if __name__ == '__main__':
    increments = [0.005]
    for increment in increments:
        create_dataset_folder(increment)