import json
import os


def main():
    with open(os.path.expanduser('~/downloads/project-1-at-2025-02-23-15-11-9e3d51af.json')) as f:
        data = json.load(f)

    selected_data = []
    for record in data:
        if record['annotations'][0]['result'][0]['value']['choices'][0] == 'Include':
            selected_data.append(record['data']['text'])
    print(f'selected {len(selected_data)} samples')

    with open(os.path.expanduser('~/downloads/selected_data.txt'), 'w') as f:
        for text in selected_data:
            f.write(f'{text}\n')
    print('Done')


if __name__ == '__main__':
    main()
