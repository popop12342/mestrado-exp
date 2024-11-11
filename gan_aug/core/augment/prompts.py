# SUBJ prompt
subj_prompt_template = """INSTRUCTION
Here are some examples of movie reviews labeled as either "Subjective" or "Objective."
Generate a new movie review for each category, keeping the tone and style similar to the examples.

EXAMPLES
{examples_text}

OUTPUT
Generate {num} new reviews that matches these classifications.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label
"""

# AclIMDB prompt
aclimdb_prompt_template = """INSTRUCTION
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

# Olist prompt
olist_prompt_template = """INSTRUCTION
Here are some examples of product reviews labeled as either "negative" or "positive."
Generate a new product review for each category, keeping the tone and style similar to the examples.

EXAMPLES
{examples_text}

OUTPUT
Generate {num} new reviews that matches these classifications.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label
"""

dataset_prompt_templates = {
    'subj': subj_prompt_template,
    'aclimdb': aclimdb_prompt_template,
    'olist': olist_prompt_template
}


def get_prompt_template(base_dataset: str) -> str:
    if base_dataset in dataset_prompt_templates:
        return dataset_prompt_templates[base_dataset]

    raise KeyError('No prompt template found for dataset ' + base_dataset)
