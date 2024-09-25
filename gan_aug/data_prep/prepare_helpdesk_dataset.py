import csv
import os

TRAIN_SPLIT = 0.8


def load_data(input_file):
    texts = []
    labels = []
    with open(input_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            texts.append(row['body'])
            labels.append(row['queue'])
    return texts, labels


def prepare_dataset(input_file, out_dir):
    texts, labels = load_data(input_file)
    print('Dataset size: ' + str(len(texts)))

    num_train = int(TRAIN_SPLIT * len(texts))
    train_texts = texts[:num_train]
    train_labels = labels[:num_train]
    test_texts = texts[num_train:]
    test_labels = labels[num_train:]
    print('Train size: ' + str(len(train_texts)))
    print('Test size: ' + str(len(test_texts)))

    train_file = os.path.join(out_dir, 'helpdesk-train.csv')
    test_file = os.path.join(out_dir, 'helpdesk-test.csv')
    write_file(train_file, train_texts, train_labels)
    write_file(test_file, test_texts, test_labels)


def write_file(csv_file, texts, labels):
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        for text, label in zip(texts, labels):
            writer.writerow({'text': text, 'label': label})


if __name__ == '__main__':
    prepare_dataset('../data/helpdesk/helpdesk_customer_tickets.csv', '../data/helpdesk')
