import os
import json
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from tqdm import tqdm
import gan_textgen_transformer
from gan_textgen_bert import objective
from bert import train

datasets = [
    'subj_001', 'subj_005', 'subj_010', 'subj_020', 'subj_030', 'subj_040',
    'subj_050', 'subj_060', 'subj_070', 'subj_080', 'subj_090', 'subj'
]


class SingleTrial:
    def __init__(self, study, trial_id):
        self.study = study
        self.trial_id = trial_id
        self.number = trial_id
        self.accuracy = 0
        self.step = 0

    def report(self, accuracy, step):
        if accuracy > self.accuracy:
            self.accuracy = accuracy
            self.step = step

    def to_dict(self):
        return {'accuracy': self.accuracy, 'step': self.step}


def run_one_experiment(dataset: str, trial_id: int, model: str) -> SingleTrial:
    Study = namedtuple('Study', ['user_attrs', 'study_name'])
    study = Study(defaultdict(str), study_name=f'{model}-llm')
    study.user_attrs['dataset'] = dataset
    study.user_attrs['num_layers'] = 1
    study.user_attrs['num_aug'] = 0
    study.user_attrs['train_aug'] = 0
    study.user_attrs['hidden_size'] = 1
    trial = SingleTrial(study, trial_id=trial_id)

    if model == 'gan-textgen-bert':
        # GAN-TEXTGEN-BERT objective
        objective(trial)
    elif model == 'gan-textgen-transformer':
        gan_textgen_transformer.objective(trial)
    elif model == 'bert':
        # simple BERT training
        train(trial)
    else:
        raise ValueError('Model is not known')
    return trial


def run_compare_experiments():
    results = {'gan-textgen-bert': {}, 'gen-bert': {}}
    trial_count = 0
    for ds in datasets:
        acc = run_one_experiment(ds, trial_count)
        trial_count += 1
        results['gan-textgen-bert'][ds] = acc
        print('Endend experiment for dataset ' + ds + ' with accuracy ' + str(acc))

    with open('stats/subj/gan-textgen-bert/compare.json', 'w') as f:
        json.dump(results, f)


def run_multiple_times_single_experiment(dataset: str, n: int, model: str):
    results = {dataset: []}
    for i in tqdm(range(n)):
        # print(f'====== Run {i+1} / {n} ======')
        trial = run_one_experiment(dataset, i, model)
        print(f'Best accuracy was {trial.accuracy} on epoch {trial.step}')
        results[dataset].append(trial.to_dict())

    stats_dir = f'stats/llmsubj/samples-20_naug-11/{model}'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    with open(f'{stats_dir}/{dataset}.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to run mutltiple times')
    parser.add_argument('--repeat', type=int, help='Number of times to repeat each experiment', default=10)
    parser.add_argument('--model', type=str, help='Model evaluated (gan-textgen-bert or bert)',
                        default='gan-textgen-bert', choices=['gan-textgen-bert', 'gan-textgen-transformer', 'bert'])
    args = parser.parse_args()
    run_multiple_times_single_experiment(args.dataset, args.repeat, args.model)
