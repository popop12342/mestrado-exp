from ..dataset_prompt_template import DatasetPromptTemplate

# AclIMDB prompt
aclimdb_prompt_template = """INSTRUCTION
Here are some examples of movie reviews labeled as either "Positive" or "Negative."
Generate a new movie reviews, keeping the tone and style similar to the examples.
{}

EXAMPLES
{{examples_text}}

OUTPUT
Generate {{num}} new reviews that matches these classifications.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label
"""

aclImdb_key_characteristics: dict[str, list[str]] = {
    'positive': [
        'the tone express enthusiasm',
        'contains positive emotional expressions',
        'has complimentary descriptions',
        'has high-quality indicators',
        'praises the acting',
        'highlights the direction',
        'appreciates the plot',
        'mentions positive feedback on music and visuals',
        'has favorable comparisons',
        'has strong recommendations',
        'mentions personal connection as a viewer',
        'mentions a memorable experience'
    ],
    'negative': [
        'tone express disappointment',
        'contains negative emotinos',
        'has critical descriptions',
        'has low-quality indicators',
        'critics the acting',
        'mentions problems with direction',
        'mentions issues in the plot',
        'has negative feedback on music and visuals',
        'has negative comparisons',
        'has warning to others',
        'has feelings of regret',
        'mentions a memorable experience for the wrong reasons'
    ]
}

aclimdb_dpt = DatasetPromptTemplate(dataset='aclImdb',
                                    prompt_template=aclimdb_prompt_template,
                                    generation_instructions=aclImdb_key_characteristics)
