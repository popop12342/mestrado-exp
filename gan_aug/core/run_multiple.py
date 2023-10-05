import json
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from gan_textgen_bert import objective

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

def run_one_experiment(dataset: str, trial_id: int) -> SingleTrial:
    Study = namedtuple('Study', ['user_attrs', 'study_name'])
    study = Study(defaultdict(str), study_name='label-vs-unlabel')
    study.user_attrs['dataset'] = dataset
    study.user_attrs['num_layers'] = 1
    study.user_attrs['num_aug'] = 0
    study.user_attrs['train_aug'] = 0
    study.user_attrs['hidden_size'] = 512
    trial = SingleTrial(study, trial_id=trial_id)
    objective(trial)
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

def run_multiple_times_single_experiment(dataset: str, n: int):
    results = {dataset: []}
    for i in range(n):
        trial = run_one_experiment(dataset, i)
        print(f'Best accuracy was {trial.accuracy} on epoch {trial.step}')
        results[dataset].append(trial.to_dict())

    with open(f'stats/subj/gan-textgen-bert/{dataset}.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to run mutltiple times')
    parser.add_argument('--repeat', type=int, help='Number of times to repeat each experiment', default=10)
    args = parser.parse_args()
    run_multiple_times_single_experiment(args.dataset, args.repeat)