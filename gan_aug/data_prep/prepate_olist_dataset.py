import csv
import os

TRAIN_SPLIT = 0.9


def load_data(input_file):
    comments = []
    scores = []
    with open(input_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            score = int(row['review_score'])
            comment = row['review_comment_message']
            # remove reviews without comments and neutral score (=3)
            if comment and score != 3:
                comment = comment.replace('\n', ' ').replace('\t', ' ')
                comments.append(comment)
                scores.append(score)
    return comments, scores


def prepare_dataset(input_file, out_dir):
    comments, scores = load_data(input_file)
    print('Dataset size: ' + str(len(scores)))

    # convert scores to labels
    labels = ['0' if score < 3 else '1' for score in scores]

    # split train and test sets
    num_train = int(TRAIN_SPLIT * len(scores))
    train_comments = comments[:num_train]
    test_comments = comments[num_train:]
    train_labels = labels[:num_train]
    test_labels = labels[num_train:]
    print('Train size: ' + str(len(train_labels)))
    print('Test size: ' + str(len(test_labels)))

    train_file = os.path.join(out_dir, 'olist-train.tsv')
    test_file = os.path.join(out_dir, 'olist-test.tsv')
    write_file(train_file, train_comments, train_labels)
    write_file(test_file, test_comments, test_labels)


def write_file(filepath, comments, labels):
    with open(filepath, 'w') as f:
        for comment, label in zip(comments, labels):
            f.write(f'{label}\t{comment}\n')


if __name__ == '__main__':
    prepare_dataset('../data/olist/olist_order_reviews_dataset.csv', '../data/olist')
