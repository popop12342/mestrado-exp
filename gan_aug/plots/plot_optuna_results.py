import json
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import Dict, List

def load_trial_stats(trial_num: int) -> List[Dict[str, object]]:
    filename = 'study-gantext-trial-{}.json'.format(trial_num)
    filepath = os.path.join('..', 'optuna', 'stats', filename)
    with open(filepath, 'r') as stat_file:
        stats = json.load(stat_file)
    return stats

def plot_train_losses(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    gen_loss = [x['Training Loss generator'] for x in trial_stats]
    dis_loss = [x['Training Loss discriminator'] for x in trial_stats]

    plt.plot(epochs, gen_loss)
    plt.plot(epochs, dis_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Generator', 'Discriminator'])

    plt.show()

def plot_loss(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    loss = [x['Valid. Loss'] for x in trial_stats]

    plt.plot(epochs, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')

    plt.show()

def plot_accuracy(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    accs = [x['Valid. Accur.'] for x in trial_stats]

    plt.plot(epochs, accs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()

def plot_f1(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    f1 = [x['Valid. F1'] for x in trial_stats]

    plt.plot(epochs, f1)
    plt.xlabel('Epochs')
    plt.ylabel('F1')

    plt.show()

def plot_recall(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    recall = [x['Valid. Recall'] for x in trial_stats]

    plt.plot(epochs, recall)
    plt.xlabel('Epochs')
    plt.ylabel('Recall')

    plt.show()

def plot_precision(trial_stats: List[Dict[str, object]]):
    epochs = [x['epoch'] for x in trial_stats]
    precision = [x['Valid. Precision'] for x in trial_stats]

    plt.plot(epochs, precision)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')

    plt.show()

def plot_trial_metrics(trial_num: int):
    stats = load_trial_stats(trial_num)
    plot_train_losses(stats)
    plot_loss(stats)
    plot_accuracy(stats)
    plot_f1(stats)
    plot_recall(stats)
    plot_precision(stats)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t',  '--trial_num', type=int, help='Optuna trial number to plot stats')
    args = parser.parse_args()

    plot_trial_metrics(args.trial_num)
    