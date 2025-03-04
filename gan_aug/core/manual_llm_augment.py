import json
import logging
import os
import random

from augment.augmentation_config import AugmentationConfig
from llm_augment import (check_dir, clean_sentences, create_llm_prompt,
                         export_to_file, load_prepare_data)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config: AugmentationConfig):
    train_data, _ = load_prepare_data(config.labels, config.dataset)

    base_dataset = config.get_base_dataset()
    prompt = create_llm_prompt(config.generate_per_round, base_dataset)

    for i in range(config.rounds):
        print(f"\nRound {i+1}/{config.rounds}\n")
        samples = random.sample(train_data, config.samples_per_round)

        examples_str = ''
        for sentence, label in samples:
            examples_str += f'Classification: {label}\n Text: {sentence}\n\n'

        prompt_value = prompt.invoke({'examples_text': examples_str})
        prompt_str = '\n\n'.join(msg.content for msg in prompt_value.to_messages())
        print(prompt_str)
        input("Press Enter to continue...")


def create_dataset_file(config: AugmentationConfig, llm_result_dir: str):
    train_data, test_data = load_prepare_data(config.labels, config.dataset)

    generated_samples: list[tuple[str, str]] = []
    for filename in os.listdir(llm_result_dir):
        if filename.endswith(".json"):
            json_file = os.path.join(llm_result_dir, filename)
            log.info(f'Loading file {json_file}')
            with open(json_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    generated_samples.append((item['text'], item['label']))

    all_sentences = train_data + generated_samples
    cleaned_samples = clean_sentences(config.labels, all_sentences)

    naug = int(len(generated_samples) / len(train_data))
    output_file = config.get_output_file(naug)
    check_dir(output_file)
    export_to_file(cleaned_samples, output_file)
    config.create_parameters_file(naug)
    log.info(f'Result written to file {output_file}')

    test_file = config.get_test_file(naug)
    if not os.path.exists(test_file):
        log.info('Creating test file')
        export_to_file(test_data, test_file)


if __name__ == '__main__':
    config = AugmentationConfig(
        dataset='helpdesk',
        labels=["General Inquiry", "Human Resources", "Billing and Payments", "Sales and Pre-Sales", "IT Support", "Customer Service", "Product Support", "Returns and Exchanges", "Service Outages and Maintenance", "Technical Support"],
        rounds=25,
        samples_per_round=5,
        generate_per_round=10,
        model='gpt-4o',
        model_type='openai'
    )
    main(config)
    # create_dataset_file(config, llm_result_dir='../data/manual_llm/r1')
