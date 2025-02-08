import logging
import os
import random
import json
from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from augment.prompts import get_prompt_template
from dataset_loader.dataset_loader import load_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model = 'gpt-4o-mini'
temperature = 0.7
llm = ChatOpenAI(
    model=model,
    temperature=temperature,
    api_key=OPENAI_API_KEY
)

DATA_DIR = '../data'

EMBEDDING_FILTER_MODEL = 'sentence-transformers/LaBSE'

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
    # print(result)
    generated_samples = []
    for record in result:
        if 'text' in record and 'label' in record:
            generated = (record['text'], record['label'])
            generated_samples.append(generated)
        else:
            log.warning('Invalid record: %s', record)
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


def get_parameters_file(dataset: str, samples: int, naug: int) -> str:
    llm_dataset = get_llm_dataset_dir_path(dataset, samples, naug)
    return os.path.join(llm_dataset, 'parameters.json')


def get_test_file(dataset: str, samples: int, naug: int) -> str:
    llm_dataset = get_llm_dataset_dir_path(dataset, samples, naug)
    return os.path.join(llm_dataset, 'test.txt')


def run_augmentation(
        labels: list[str],
        dataset: str,
        rounds: int,
        samples_per_round: int,
        generated_per_round: int,
        threshold: float,
        filter: bool):
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset)
    train_labels_str = [labels[int(y)] for y in train_labels]

    train_data = list(zip(train_sentences, train_labels_str))

    base_dataset = get_base_dataset(dataset)
    generated_samples = []
    with get_openai_callback() as openai_cb:
        for _ in tqdm(range(rounds), desc='Gen round'):
            # start_idx = i * samples_per_round
            # end_idx = (i+1) * samples_per_round
            # samples = train_data[start_idx:end_idx]
            samples = random.sample(train_data, samples_per_round)
            gen_samples = augment_data(samples, generated_per_round, base_dataset)
            generated_samples.extend(gen_samples)
    log.info(openai_cb)

    log.info(f'Generated {len(generated_samples)} samples')

    if filter:
        log.info('Filtering generated samples')
        generated_samples = embedding_similarity_filter(train_data, generated_samples, threshold)
        log.info(f'Filtered {len(generated_samples)} samples')
    all_sentences = train_data + generated_samples

    log.info(f'Total samples: {len(all_sentences)}')

    cleaned_samples = []
    for sentence, label in all_sentences:
        label_id = labels.index(label.lower())
        sentence = sentence.strip()
        cleaned_samples.append((sentence, label_id))

    naug = int(generated_per_round * rounds / len(train_data))
    output_file = get_output_file(dataset, samples_per_round, naug)
    check_dir(output_file)
    export_to_file(cleaned_samples, output_file)
    create_parameters_file(dataset, rounds, samples_per_round, generated_per_round, naug, threshold, filter)
    log.info(f'Result wrote to file {output_file}')

    test_data = list(zip(test_sentences, test_labels))
    test_file = get_test_file(dataset, samples_per_round, naug)
    if not os.path.exists(test_file):
        log.info('Creating test file')
        export_to_file(test_data, test_file)


def check_dir(output_file):
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        log.info('LLM dataset directory does not exists, creating it')
        os.mkdir(dirname)


def create_parameters_file(dataset: str, rounds: int, sample_per_round: int, generated_per_round: int,
                           naug: int, threshold: float, filter: bool):
    parameters_file = get_parameters_file(dataset, sample_per_round, naug)
    parameters = {
        'dataset': dataset,
        'rounds': rounds,
        'samples_per_round': sample_per_round,
        'generated_per_round': generated_per_round,
        'naug': naug,
        'threshold': threshold,
        'filter': filter,
        'model': model,
        'temperature': temperature
    }
    with open(parameters_file, 'w') as f:
        json.dump(parameters, f)
    log.info(f'Parameters wrote to file {parameters_file}')


def embedding_similarity_filter(
        real_samples: list[tuple[str, str]],
        synthetic_samples: list[tuple[str, str]],
        threshold: float = 0.8) -> list[str]:
    embedder = SentenceTransformer(EMBEDDING_FILTER_MODEL)
    real_texts = [text for text, _ in real_samples]
    synthetic_texts = [text for text, _ in synthetic_samples]
    real_embeddings = embedder.encode(real_texts, convert_to_tensor=True)
    synthetic_embeddings = embedder.encode(synthetic_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(synthetic_embeddings, real_embeddings)
    filtered_samples = [
        sample for idx, sample in enumerate(synthetic_samples) if cosine_scores[idx].max() > threshold
    ]
    return filtered_samples


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels', nargs=2, default=['subjective', 'objective'])  # test for binary classification
    parser.add_argument('--dataset', default='subj_001')
    parser.add_argument('--rounds', default=1, type=int)
    parser.add_argument('--samples_per_round', default=5, type=int)
    parser.add_argument('--generated_per_round', default=5, type=int)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--filter', default=False, type=bool)

    args = parser.parse_args()
    run_augmentation(
        labels=args.labels,
        dataset=args.dataset,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        generated_per_round=args.generated_per_round,
        threshold=args.threshold,
        filter=args.filter
    )
