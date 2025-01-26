from ..dataset_prompt_template import DatasetPromptTemplate

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

subj_dpt = DatasetPromptTemplate(dataset='subj', prompt_template=subj_prompt_template)
