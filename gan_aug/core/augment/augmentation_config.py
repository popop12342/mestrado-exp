import os
import json
import logging
from pydantic import BaseModel, Field


log = logging.getLogger(__name__)

DATA_DIR = '../data'


class AugmentationConfig(BaseModel):
    dataset: str
    labels: list[str]
    rounds: int
    samples_per_round: int
    generate_per_round: int
    threshold: float = Field(default=0.0)
    filter_enabled: bool = Field(default=False)
    model: str = Field(default='gpt-4o-mini')
    model_type: str = Field(default='openai')

    def create_parameters_file(self, naug: int):
        parameters_file = self.get_parameters_file(naug)
        parameters = {
            'dataset': self.dataset,
            'rounds': self.rounds,
            'samples_per_round': self.samples_per_round,
            'generated_per_round': self.generate_per_round,
            'naug': naug,
            'threshold': self.threshold,
            'filter': self.filter_enabled,
            'model': self.model
        }
        with open(parameters_file, 'w') as f:
            json.dump(parameters, f)
        log.info(f'Parameters wrote to file {parameters_file}')

    def get_llm_dataset_dir_path(self, naug: int) -> str:
        base_dataset = self.get_base_dataset()
        llm_dataset = 'llm' + base_dataset
        augmentation_dir_name = f'samples-{self.samples_per_round}_naug-{naug}'
        return os.path.join(DATA_DIR, llm_dataset, augmentation_dir_name)

    def get_base_dataset(self) -> str:
        base_dataset = self.dataset
        if '_' in self.dataset:
            base_dataset = self.dataset.split('_')[0]
        return base_dataset

    def get_output_file(self, naug: int) -> str:
        llm_dataset = self.get_llm_dataset_dir_path(naug)
        filename = f'llm_{self.dataset}.txt'
        return os.path.join(llm_dataset, filename)

    def get_parameters_file(self, naug: int) -> str:
        llm_dataset = self.get_llm_dataset_dir_path(naug)
        return os.path.join(llm_dataset, 'parameters.json')

    def get_test_file(self, naug: int) -> str:
        llm_dataset = self.get_llm_dataset_dir_path(naug)
        return os.path.join(llm_dataset, 'test.txt')
