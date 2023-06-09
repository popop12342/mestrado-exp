import os
import random

def create_dataset(increment):
    lines = open('data/subj/train_orig.txt').readlines()
    train_size = int(len(lines) * increment)
    random.shuffle(lines)
    lines = lines[:train_size]

    filename = 'train_{:03d}.txt'.format(int(100 * increment))
    with open(os.path.join('data/subj', filename), 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    create_dataset(0.05)
