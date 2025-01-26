from .dataset_prompt_template import DatasetPromptTemplate
from .prompt_templates.aclimdb_prompt_template import aclimdb_dpt
from .prompt_templates.olist_prompt_template import olist_dpt
from .prompt_templates.rotten400k_prompt_template import rotten400k_dpt
from .prompt_templates.subj_prompt_template import subj_dpt


prompt_template_registry: dict[str, DatasetPromptTemplate] = {
    'subj': subj_dpt,
    'aclImdb': aclimdb_dpt,
    'rotten400k': rotten400k_dpt,
    'olist': olist_dpt
}


def get_prompt_template(base_dataset: str) -> str:
    if base_dataset in prompt_template_registry:
        prompt_template = prompt_template_registry[base_dataset]
        return prompt_template.build_template()

    raise KeyError('No prompt template found for dataset ' + base_dataset)
