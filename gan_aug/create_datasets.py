import os
import random

def create_dataset(increment: float):
    lines = open('data/subj/train_orig.txt').readlines()
    train_size = int(len(lines) * increment)
    random.shuffle(lines)
    lines = lines[:train_size]

    filename = 'train_{:03d}.txt'.format(int(100 * increment))
    with open(os.path.join('data/subj', filename), 'w') as f:
        f.writelines(lines)

def create_unlabeled_dataset(increment: float):
    lines = open('data/subj/train_orig.txt').readlines()
    train_size = int(len(lines) * increment)
    random.shuffle(lines)
    train_lines = lines[:train_size]
    
    unlabeled_lines = []
    for line in lines[train_size:]:
        _, sentence = line.split('\t')
        unlabeled_lines.append('UNK\t' + sentence)

    filename = 'train_un_{:03d}.txt'.format(int(100 * increment))
    with open(os.path.join('data/subj', filename), 'w') as f:
        f.writelines(train_lines)
        f.writelines(unlabeled_lines)

if __name__ == '__main__':
    create_unlabeled_dataset(0.05)
