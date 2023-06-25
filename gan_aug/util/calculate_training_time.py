import json

def calculate(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    time = 0
    for s in stats:
        timestamp = s['Training Time']
        hours, minutes, seconds = timestamp.split(':')
        time += int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    hours = time // 3600
    minutes = (time % 3600) // 60
    seconds = time % 60
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

if __name__ == '__main__':
    print(calculate('optuna/stats/subj/word2vec/study-subj-1layer-trial-0.json'))