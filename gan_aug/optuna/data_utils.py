import datetime
import json
import os
from typing import Dict, List

from optuna.trial import Trial


def save_stats(stats: List[Dict], trial: Trial):
    filename = 'stats/study-{}-trial-{}.json'.format(trial.study.study_name, trial.number)
    with open(filename, 'w') as json_file:
        json.dump(stats, json_file)
    gdrive_path = '/gdrive/MyDrive/Mestrado'
    if os.path.exists(gdrive_path):
        with open(os.path.join(gdrive_path, filename), 'w') as json_file:
            json.dump(stats, json_file)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))