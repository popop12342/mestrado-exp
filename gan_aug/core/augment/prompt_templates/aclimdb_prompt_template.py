from ..dataset_prompt_template import DatasetPromptTemplate

# AclIMDB prompt
aclimdb_prompt_template = """INSTRUCTION
Here are some examples of movie reviews labeled as either "Positive" or "Negative."
Generate a new movie reviews, keeping the tone and style similar to the examples.
These reviews are good because they are representative of movie lovers who engage in discussions about movies in
websites like IMDB, and shows the common ways people express their opinions about movies and their aspects.
EXAMPLES

{examples_text}

OUTPUT
Generate {num} new reviews, their classification and a reason why that is a good sentence
to include in the training dataset.
Your output should be in JSON format and you shoud ONLY return this JSON.
Return a list of objects with the keys `text` (for the generated text),
`label` for its label, and `reason` for the reason why that is a good sentence.
"""

# aclimdb_prompt_template = """INSTRUCTION
# In a movie reviews dataset each sample is labeled as either "Positive" or "Negative."
# Generate a new movie reviews from the IMDB website with varied style and tone.
# The reviews could be generic about the entire movie or could be specific to single
# aspects about the movie. You can include the name of the movie, actors or characters
# in the generated review. Make them sound like they were written by movie loves
# who love to engage in discussions about movies.

# OUTPUT
# Generate {num} new reviews that matches these classifications.
# Your output should be in JSON format and you shoud onyl return this JSON.
# Return a list of objects with the keys `text` (for the generated text) and
# `label` for its label
# """

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
        'contains negative emotions',
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
                                    prompt_template=aclimdb_prompt_template)
                                    # generation_instructions=aclImdb_key_characteristics)
