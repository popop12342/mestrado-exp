from .aclimdb_prompt_template import aclimdb_prompt_template, aclImdb_key_characteristics
from ..dataset_prompt_template import DatasetPromptTemplate

rotten400k_dpt = DatasetPromptTemplate(dataset='rotten400k',
                                       prompt_template=aclimdb_prompt_template,
                                       generation_instructions=aclImdb_key_characteristics)
