import os
import random

import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def transform_lines(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            label, sentence = line.split('\t')
            sentence = get_only_chars(sentence)
            lines.append('SUBJ:' + label + ' ' + sentence + '\n')
    return lines
    

def create_dataset_folder(increment):
    train_lines = transform_lines('data/subj/train_orig.txt')
    test_lines = transform_lines('data/subj/test.txt')

    dir_name = 'data/subj_{:03d}'.format(int(100 * increment))
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

def create_unlabeled_dataset(increment):
    train_lines = transform_lines('data/subj/train_orig.txt')
    test_lines = transform_lines('data/subj/test.txt')

    dir_name = 'data/subj_un_{:03d}'.format(int(100 * increment))
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    train_size = int(len(train_lines) * increment)
    random.shuffle(train_lines)
    unlabel_train_lines = train_lines[train_size:]
    train_lines = train_lines[:train_size]
    for i in range(len(unlabel_train_lines)):
        line = unlabel_train_lines[i].split()
        unlabel_train_lines[i] = 'UNK:UNK ' + ' '.join(line[1:]) + '\n'

    with open(os.path.join(dir_name, 'labeled.tsv'), 'w') as f:
        f.write('fine_label utterance\n')
        f.writelines(train_lines)

    with open(os.path.join(dir_name, 'test.tsv'), 'w') as f:
        f.write('fine_label utterance\n')
        f.writelines(test_lines)
    
    with open(os.path.join(dir_name, 'unlabeled.tsv'), 'w') as f:
        f.write('fine_label utterance\n')
        f.writelines(unlabel_train_lines)

if __name__ == '__main__':
    increments = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for increment in increments:
        create_dataset_folder(increment)
    for increment in increments:
        create_unlabeled_dataset(increment)