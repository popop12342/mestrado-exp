import logging
import os
import random
from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.language_models import BaseChatModel
from langchain_community.callbacks.manager import get_openai_callback
from tqdm import tqdm

from augment.prompts import get_prompt_template
from augment.augmentation_config import AugmentationConfig
from augment.filter.filter_augmentation import filter_augmentation, llm_filter
from dataset_loader.dataset_loader import load_dataset, get_label_names

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


DATA_DIR = '../data'


# SYSTEM_PROMPT = "You are a text generation expert, capable of producing novel high quality labeled examples. You have\
# a very good understanding of the language and the specific domain you are requested. The texts you produce are similar\
# to the ones you recieve as an example and are variend in term of vocabulary used, expressions, manners, delivery, etc.\
# You always produced examples labeled correctly."

SYSTEM_PROMPT = """You are a text generation expert specializing in creating high-quality, novel labeled examples for text classification tasks. You have a deep understanding of language nuances and the specific domain provided. Your task is to generate text that closely aligns with the examples given, while introducing diversity in vocabulary, expressions, tone, style, and delivery.

Each example must be accurately labeled and exhibit clear alignment with the provided label's characteristics. For subjective texts, focus on personal opinions, emotions, and impressions. For objective texts, maintain factual and neutral descriptions. Similarly, ensure sentiment labels (positive/negative) match the tone and content of the text.

Your output must be coherent, varied, and domain-appropriate to enhance the dataset effectively. Avoid repeating patterns or introducing biases inconsistent with the provided examples."""


def create_chat_model(model: str = 'gpt-4o-mini', model_type='openai', **kwargs) -> BaseChatModel:
    if model_type == 'openai':
        return ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            **kwargs
        )
    elif model_type == 'huggingface':
        llm = HuggingFacePipeline.from_model_id(
            model_id=model,
            task='text-generation',
            pipeline_kwargs={
                'max_new_tokens': 5000
            })
        llm.pipeline.tokenizer.pad_token_id = 0

        return llm
    elif model_type == 'huggingface-endpoint':
        llm = HuggingFaceEndpoint(
            repo_id=model,
            max_new_tokens=10_000
        )
        return ChatHuggingFace(llm=llm)
    log.error('Invalid model type: %s', model_type)
    raise ValueError('Invalid model type')


def augment_data(samples: list[tuple[str, str]], generated_per_round: int, base_dataset: str, llm: BaseChatModel) -> list[tuple[str, str]]:
    prompt = create_llm_prompt(generated_per_round, base_dataset)
    pipeline = prompt | llm | JsonOutputParser()

    examples_str = ''
    for sentence, label in samples:
        examples_str += f'Classification: {label}\n Review: {sentence}\n\n'

    result = None
    retry = 0
    max_retry = 3
    while not result and retry < max_retry:
        try:
            result = pipeline.invoke({'examples_text': examples_str})
        except Exception as e:
            retry += 1
            log.warning(f'Failed invoking generation pipeline with error {str(e)}, retry {retry}')

    generated_samples = []
    if result:
        log.debug(result)
        for record in result:
            if 'text' in record and 'label' in record:
                generated = (record['text'], record['label'])
                generated_samples.append(generated)
            else:
                log.warning('Invalid record: %s', record)
    return generated_samples


def create_llm_prompt(generated_per_round: int, base_dataset: str) -> ChatPromptTemplate:
    prompt_template = get_prompt_template(base_dataset=base_dataset)
    prompt = ChatPromptTemplate([
        ('system', SYSTEM_PROMPT),
        ('user', prompt_template)
    ], partial_variables={'num': str(generated_per_round)})

    return prompt


def export_to_file(samples: list[tuple[str, str]], output_file: str):
    with open(output_file, 'w') as f:
        for sentence, label in samples:
            sentence = sentence.replace('\n', '\\n')
            f.write(f'{label}\t{sentence}\n')


def run_augmentation(config: AugmentationConfig):
    labels = get_label_names(config.dataset)
    log.info('Labels: %s', labels)
    train_data, test_data = load_prepare_data(labels, config.dataset)

    llm = create_chat_model(config.model, config.model_type)

    base_dataset = config.get_base_dataset()
    generated_samples = []
    with get_openai_callback() as openai_cb:
        for _ in tqdm(range(config.rounds), desc='Gen round'):
            # start_idx = i * samples_per_round
            # end_idx = (i+1) * samples_per_round
            # samples = train_data[start_idx:end_idx]
            samples = random.sample(train_data, config.samples_per_round)
            gen_samples = augment_data(samples, config.generate_per_round, base_dataset, llm)
            generated_samples.extend(gen_samples)
    log.info(openai_cb)

    log.info(f'Generated {len(generated_samples)} samples')

    if config.filter_enabled:
        log.info('Filtering generated samples')
        generated_samples = filter_augmentation(train_data, generated_samples, config.threshold)
        log.info(f'Filtered {len(generated_samples)} samples')
    all_sentences = train_data + generated_samples

    log.info(f'Total samples: {len(all_sentences)}')

    cleaned_samples = clean_sentences(labels, all_sentences)

    naug = int(config.generate_per_round * config.rounds / len(train_data))
    output_file = config.get_output_file(naug)
    check_dir(output_file)
    export_to_file(cleaned_samples, output_file)
    config.create_parameters_file(naug)
    log.info(f'Result wrote to file {output_file}')

    test_file = config.get_test_file(naug)
    if not os.path.exists(test_file):
        log.info('Creating test file')
        export_to_file(test_data, test_file)


def clean_sentences(labels: list[str], all_sentences: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned_samples = []
    lower_labels = [label.lower() for label in labels]
    log.info('Lower labels %s', lower_labels)
    for sentence, label in all_sentences:
        if label.lower() not in lower_labels:
            log.error('Invalid label: %s; %s', label, sentence)
            continue
        label_id = lower_labels.index(label.lower())
        sentence = sentence.strip()
        cleaned_samples.append((sentence, label_id))
    return cleaned_samples


def load_prepare_data(labels: list[str], dataset: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    log.info('Loading dataset %s', dataset)
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(dataset)
    train_labels_str = [labels[int(y)] for y in train_labels]

    train_data = list(zip(train_sentences, train_labels_str))
    test_data = list(zip(test_sentences, test_labels))
    return train_data, test_data


def check_dir(output_file):
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        log.info('LLM dataset directory does not exists, creating it')
        os.mkdir(dirname)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--labels', nargs=2, default=['subjective', 'objective'])  # test for binary classification
    parser.add_argument('--dataset', default='subj_001')
    parser.add_argument('--rounds', default=1, type=int)
    parser.add_argument('--samples_per_round', default=5, type=int)
    parser.add_argument('--generated_per_round', default=5, type=int)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--filter', default=False, type=bool)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--model_type', default='openai')

    args = parser.parse_args()
    config = AugmentationConfig(
        dataset=args.dataset,
        # labels=args.labels,
        rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        generate_per_round=args.generated_per_round,
        threshold=args.threshold,
        filter_enabled=args.filter,
        model=args.model,
        model_type=args.model_type
    )
    run_augmentation(config)

    # Just for testing llm filtering, using the same real dataset and reference and to filter
    # real_sentences, real_labels, _, _ = load_dataset('aclImdb_001')
    # real_data = list(zip(real_sentences, real_labels))
    # fake_sentences, fake_labels, _, _ = load_dataset('cllmaclImdb_001')
    # fake_data = list(zip(fake_sentences, fake_labels))
    # fake_data = fake_data[len(real_data):]
    # print(f'only filtering {len(fake_data)} fake samples')

    # filtered_samples = llm_filter(real_data, fake_data)
    # print(f'Filtered {len(filtered_samples)} samples')
    # export_to_file(filtered_samples, '../data/aclImdb_001_filtered.txt')
