import random
import os
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from ..augmentation_config import AugmentationConfig

EMBEDDING_FILTER_MODEL = 'sentence-transformers/LaBSE'


def filter_augmentation(
        real_samples: list[tuple[str, str]],
        synthetic_samples: list[tuple[str, str]],
        config: AugmentationConfig) -> list[tuple[str, str]]:
    if config.filter_kind == 'similarity':
        return embedding_similarity_filter(real_samples, synthetic_samples, config.threshold)
    elif config.filter_kind == 'llm':
        return llm_filter(real_samples, synthetic_samples, config.threshold)
    else:
        raise ValueError('Invalid filter kind')


def embedding_similarity_filter(
        real_samples: list[tuple[str, str]],
        synthetic_samples: list[tuple[str, str]],
        threshold: float = 0.8) -> list[tuple[str, str]]:
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


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REAL_SAMPLES_SIZE = 20
SYNTHETIC_SAMPLES_SIZE = 20

SYSTEM_PROMPT = """You are a text reviewers specialist. You are taskted with reviewing texts writted by a language model and evaluating its quality and similarity from a base reference dataset.
For each new text, you will give a score from 0 to 1, where 0 is the worst possible score and 1 is the best possible score.
Bad texts are one that are not natural, are not coherent, or are not similar to the base reference dataset or are just too similar with only minor variation.
Good texts are ones that are natural, coherent, and could be find in the same contexts of the reference base dataset.
"""

USER_PROMPT = """Review the following text and give a score from 0 to 1. 0 is the worst possible score and 1 is the best possible score.

REFERENCE TEXTS:
{reference_texts}

NEW TEXT:
{new_text}

OUTPUT
For each new text, you will give a score from 0 to 1, where 0 is the worst possible score and 1 is the best possible score.
Your ourput should be in JSON format and you should only return this JSON.
Return a list of objects with the keys `score` (for the generated text), `text` for the evaluated text and `label` for its label.
"""


def llm_filter(
        real_samples: list[tuple[str, str]],
        synthetic_samples: list[tuple[str, str]],
        threshold: float = 0.5) -> list[tuple[str, str]]:
    prompt = ChatPromptTemplate([
        ('system', SYSTEM_PROMPT),
        ('user', USER_PROMPT)
    ])
    llm = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY)
    pipe = prompt | llm | JsonOutputParser()
    filtered_samples = []
    batch = SYNTHETIC_SAMPLES_SIZE
    turns = len(synthetic_samples) // batch
    for i in tqdm(range(turns), desc='Filtering samples'):
        samples = random.sample(real_samples, REAL_SAMPLES_SIZE)
        reference_texts = '\n'.join(f'label: {label} - text:{text}' for text, label in samples)
        batch_synthetic_samples = synthetic_samples[i*batch:(i+1)*batch]
        new_texts = '\n'.join(f'label: {label} - text:{text}' for text, label in batch_synthetic_samples)
        result = pipe.invoke({'reference_texts': reference_texts, 'new_text': new_texts})
        for record in result:
            if 'score' in record and 'text' in record and 'label' in record:
                if record['score'] > threshold:
                    filtered_samples.append((record['text'], record['label']))
    return filtered_samples


if __name__ == '__main__':
    # Just for testing llm filtering, using the same real dataset and reference and to filter
    from ...dataset_loader.dataset_loader import load_dataset
    train_sentences, train_labels, test_sentences, test_labels = load_dataset('aclImdb_001')
    train_data = list(zip(train_sentences, train_labels))
    filtered_samples = llm_filter(train_data, train_data)
    for text, label in filtered_samples:
        print(f'{label}\t{text}')
    print(f'Filtered {len(filtered_samples)} samples')
