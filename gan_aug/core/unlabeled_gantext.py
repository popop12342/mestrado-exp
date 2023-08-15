from collections import namedtuple, defaultdict
from gan_optuna import objective
import json

datasets = [
    'subj_001', 'subj_005', 'subj_010', 'subj_020', 'subj_030', 'subj_040', 
    'subj_050', 'subj_060', 'subj_070', 'subj_080', 'subj_090', 'subj'
]

unlabeled_dataset = [
    'subj_un_001', 'subj_un_005', 'subj_un_010', 'subj_un_020', 'subj_un_030',
    'subj_un_040', 'subj_un_050', 'subj_un_060', 'subj_un_070', 'subj_un_080',
    'subj_un_090', 'subj'
]

class SingleTrial:
    def __init__(self, study, trial_id):
        self.study = study
        self.trial_id = trial_id
        self.number = trial_id

    def report(self, accuracy, step):
        pass

def run_one_experiment(dataset: str, trial_id: int) -> float:
    Study = namedtuple('Study', ['user_attrs', 'study_name'])
    study = Study(defaultdict(str), study_name='label-vs-unlabel')
    study.user_attrs['dataset'] = dataset
    study.user_attrs['num_layers'] = 1
    study.user_attrs['num_aug'] = 0
    study.user_attrs['train_aug'] = 0
    study.user_attrs['hidden_size'] = 512
    trial = SingleTrial(study, trial_id=trial_id)
    return objective(trial)

def run_experiments():
    results = {'labeled': {}, 'unlabeled': {}}
    trial_count = 0
    for ds in unlabeled_dataset:
        acc = run_one_experiment(ds, trial_count)
        trial_count += 1
        results['unlabeled'][ds] = acc
        print('Endend experiment for dataset ' + ds + ' with accuracy ' + str(acc))
    for ds in datasets:
        acc = run_one_experiment(ds, trial_count)
        trial_count += 1
        results['labeled'][ds] = acc
        print('Endend experiment for dataset ' + ds + ' with accuracy ' + str(acc))
    
    with open('stats/subj/unlabeled/label-vs-unlabel-2.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    run_experiments()