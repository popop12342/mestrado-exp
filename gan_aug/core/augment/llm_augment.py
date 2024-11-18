import os
import sys
import random
from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from tqdm import tqdm

from prompts import get_prompt_template
sys.path.append('.')
from dataset_loader.dataset_loader import load_dataset

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)

DATA_DIR = '../data'

# SYSTEM_PROMPT = "You are a text generation expert, capable of producing novel high quality labeled examples. You have\
# a very good understanding of the language and the specific domain you are requested. The texts you produce are similar\
# to the ones you recieve as an example and are variend in term of vocabulary used, expressions, manners, delivery, etc.\
# You always produced examples labeled correctly."

SYSTEM_PROMPT = """You are a text generation expert specializing in creating high-quality, novel labeled examples for text classification tasks. You have a deep understanding of language nuances and the specific domain provided. Your task is to generate text that closely aligns with the examples given, while introducing diversity in vocabulary, expressions, tone, style, and delivery.

Each example must be accurately labeled and exhibit clear alignment with the provided label's characteristics. For subjective texts, focus on personal opinions, emotions, and impressions. For objective texts, maintain factual and neutral descriptions. Similarly, ensure sentiment labels (positive/negative) match the tone and content of the text.

Your output must be coherent, varied, and domain-appropriate to enhance the dataset effectively. Avoid repeating patterns or introducing biases inconsistent with the provided examples."""

def augment_data(samples: list[tuple[str, str]], generated_per_round: int, base_dataset: str) -> list[tuple[str, str]]:
    prompt_template = get_prompt_template(base_dataset=base_dataset)
    prompt = ChatPromptTemplate([
        ('system', SYSTEM_PROMPT),
        ('user', prompt_template)
    ], partial_variables={'num': str(generated_per_round)})
    pipeline = prompt | llm | JsonOutputParser()

    examples_str = ''
    for sentence, label in samples:
        examples_str += f'Classification: {label}\n Review: {sentence}\n\n'

    # print(prompt.invoke({'examples_text': examples_str}))
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


def get_llm_dataset_dir_path(dataset: str, samples: int, naug: int) -> str:
    base_dataset = get_base_dataset(dataset)
    llm_dataset = 'llm' + base_dataset
    augmentation_dir_name = f'samples-{samples}_naug-{naug}'
    return os.path.join(DATA_DIR, llm_dataset, augmentation_dir_name)


def get_base_dataset(dataset: str) -> str:
    base_dataset = dataset
    if '_' in dataset:
        base_dataset = dataset.split('_')[0]
    return base_dataset


def get_output_file(dataset: str, samples: int, naug: int) -> str:
    llm_dataset = get_llm_dataset_dir_path(dataset, samples, naug)
    filename = f'llm_{dataset}.txt'
    return os.path.join(llm_dataset, filename)


def get_test_file(dataset: str, samples: int, naug: int) -> str:
    llm_dataset = get_llm_dataset_dir_path(dataset, samples, naug)
    return os.path.join(llm_dataset, 'test.txt')


def run_augmentation(
        labels: list[str],
        dataset: str,
        rounds: int,
        samples_per_round: int,
        generated_per_round: int):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset)
    train_labels_str = [labels[int(y)] for y in train_labels]

    train_data = list(zip(train_sentences, train_labels_str))

    base_dataset = get_base_dataset(dataset)
    generated_samples = []
    with get_openai_callback() as openai_cb:
        for _ in tqdm(range(rounds)):
            # start_idx = i * samples_per_round
            # end_idx = (i+1) * samples_per_round
            # samples = train_data[start_idx:end_idx]
            samples = random.sample(train_data, samples_per_round)
            gen_samples = augment_data(samples, generated_per_round, base_dataset)
            generated_samples.extend(gen_samples)
    print(openai_cb)

    all_sentences = train_data + generated_samples

    cleaned_samples = []
    for sentence, label in all_sentences:
        label_id = labels.index(label)
        sentence = sentence.strip()
        cleaned_samples.append((sentence, label_id))

    naug = int(generated_per_round * rounds / len(train_data))
    output_file = get_output_file(dataset, samples_per_round, naug)
    check_dir(output_file)
    export_to_file(cleaned_samples, output_file)
    print(f'Result wrote to file {output_file}')

    test_data = list(zip(test_sentences, test_labels))
    test_file = get_test_file(dataset, samples_per_round, naug)
    if not os.path.exists(test_file):
        print('Creating test file')
        export_to_file(test_data, test_file)


def check_dir(output_file):
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        print('LLM dataset directory does not exists, creating it')
        os.mkdir(dirname)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels', nargs=2, default=['subjective', 'objective'])  # test for binary classification
    parser.add_argument('--dataset', default='subj_001')
    parser.add_argument('--rounds', default=1, type=int)
    parser.add_argument('--samples_per_round', default=5, type=int)
    parser.add_argument('--generated_per_round', default=5, type=int)

    args = parser.parse_args()
    run_augmentation(
        labels=args.labels,
        dataset=args.dataset,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        generated_per_round=args.generated_per_round
    )
