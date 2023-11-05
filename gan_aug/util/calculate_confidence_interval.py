import math
import json

Z = 2.685

def calculate_interval(values: list[float]) -> tuple[float, float]:
    avg = sum(values) / len(values)
    var = sum([(x-avg)**2 for x in values]) / (len(values) - 1)
    std = math.sqrt(var)
    ci = Z * std / math.sqrt(len(values))
    return avg, ci

def calculate_model_intervals(data, model_name: str):
    print('====== ' + model_name.upper()+ ' ======')
    for dataset in data[model_name]:
        print(dataset)
        if len(data[model_name][dataset]) > 1:
            accs = [x['accuracy'] for x in data[model_name][dataset]]
            avg, ci = calculate_interval(accs)
            print('{} +- {}'.format(avg, ci))
        else:
            print('Not calculated yet')

def get_avgs(data, model_name):
    avgs = []
    for dataset in data[model_name]:
        accs = [x['accuracy'] for x in data[model_name][dataset]]
        avg, _ = calculate_interval(accs)
        avgs.append(avg)
    return avgs

def calculate_gains(data):
    textgen_avg = get_avgs(data, 'gan-textgen-bert')
    bert_avg = get_avgs(data, 'gan-bert')
    print('====== GAIN (%) ======')
    for x, y in zip(textgen_avg, bert_avg):
        gain = (x - y) / y
        print(x-y, gain)

def calculate_all_intervals(file: str):
    with open(file, 'r') as f:
        data = json.load(f)
    
    calculate_model_intervals(data, 'gan-textgen-bert')
    calculate_model_intervals(data, 'gan-bert')
    calculate_gains(data)


if __name__ ==  '__main__':
    calculate_all_intervals('../core/stats/subj/gan-textgen-bert/compare-interval.json')