import os
import random
import sys
from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

sys.path.append('.')

from dataset_loader.dataset_loader import load_dataset

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)


# SUBJ prompt
# prompt_template = """INSTRUCTION
# Here are some examples of movie reviews labeled as either "Subjective" or "Objective."
# Generate a new movie review for each category, keeping the tone and style similar to the examples.

# EXAMPLES
# {examples_text}

# OUTPUT
# Generate {num} new reviews that matches these classifications.
# Your output should be in JSON format and you shoud onyl return this JSON.
# Return a list of objects with the keys `text` (for the generated text) and
# `label` for its label
# """

# AclIMDB prompt
prompt_template = """INSTRUCTION
Here are some examples of movie reviews labeled as either "Positive" or "Negative."
Generate a new movie review for each category, keeping the tone and style similar to the examples.

EXAMPLES
{examples_text}

OUTPUT
Generate {num} new reviews that matches these classifications.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label
"""


def augment_data(samples: list[tuple[str, str]], generated_per_round: int) -> list[tuple[str, str]]:

    prompt = PromptTemplate.from_template(prompt_template, partial_variables={'num': str(generated_per_round)})
    pipeline = prompt | llm | JsonOutputParser()

    examples_str = ''
    for sentence, label in samples:
        examples_str += f'Classification: {label}\n Review: {sentence}\n\n'

    result = pipeline.invoke({'examples_text': examples_str})
    generated_samples = []
    for record in result:
        generated = (record['text'], record['label'])
        generated_samples.append(generated)
    return generated_samples


def export_to_file(samples: list[tuple[str, str]], output_file: str):
    with open(output_file, 'w') as f:
        for sentence, label in samples:
            f.write(f'{label}\t{sentence}\n')
    print(f'Result wrote to file {output_file}')


def run_augmentation(
        labels: list[str],
        dataset: str,
        rounds: int,
        samples_per_round: int,
        generated_per_round: int,
        output_file: str):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset)
    train_labels_str = [labels[int(y)] for y in train_labels]

    train_data = list(zip(train_sentences, train_labels_str))

    generated_samples = []
    with get_openai_callback() as openai_cb:
        for i in range(rounds):
            start_idx = i * samples_per_round
            end_idx = (i+1) * samples_per_round
            samples = train_data[start_idx:end_idx]
            # samples = random.sample(train_data, samples_per_round)
            gen_samples = augment_data(samples, generated_per_round)
            generated_samples.extend(gen_samples)
            print(f'Round {i} done')
    print(openai_cb)

    all_sentences = train_data + generated_samples

    cleaned_samples = []
    for sentence, label in all_sentences:
        label_id = labels.index(label)
        sentence = sentence.strip()
        cleaned_samples.append((sentence, label_id))

    export_to_file(cleaned_samples, output_file)

    test_data = list(zip(test_sentences, test_labels))
    test_file = '../data/llmaclImdb/test.txt'
    if not os.path.exists(test_file):
        print('Creating test file')
        export_to_file(test_data, test_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels', nargs=2, default=['subjective', 'objective'])  # test for binary classification
    parser.add_argument('--dataset', default='subj_001')
    parser.add_argument('--rounds', default=1, type=int)
    parser.add_argument('--samples_per_round', default=5, type=int)
    parser.add_argument('--generated_per_round', default=5, type=int)
    parser.add_argument('--output_file', default='../data/llmsubj/llm_subj_001.txt')

    args = parser.parse_args()
    run_augmentation(
        labels=args.labels,
        dataset=args.dataset,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        generated_per_round=args.generated_per_round,
        output_file=args.output_file
    )
