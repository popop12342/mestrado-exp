from ..dataset_prompt_template import DatasetPromptTemplate


# Olist prompt
olist_prompt_template = """INSTRUCTION
Here are some examples of product reviews labeled as either "negative" or "positive."
Generate a new product review for each category, keeping the tone and style similar to the examples.
{}

EXAMPLES
{{examples_text}}

OUTPUT
Generate {{num}} new reviews that matches these classifications.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label
"""


olist_key_characteristics: dict[str, list[str]] = {
    'positive': [
        'the tone express satisfaction',
        'the tone express enthusiasm',
        'uses adjectives for quality and performance',
        'highlights the value for money',
        'compliment the aesthetic and appeal',
        'mentions specific functionality',
        'mentions the product durability',
        'mentions the comfort of use',
        'mentions the delivery and packaging',
        'says the product is outperforming the alternatives',
        'makes strong recommendations',
        'comments about the problem solving experience',
        'reinforce the product usefulness',
        'has gratitude or enjoyment about the product',
        'has a positive emotional response',
        'praises the brand or seller',
        'mentions intent to re-purchase'
    ],
    'negative': [
        'the tone express disappointment',
        'the tone express negative emotions',
        'uses adjectives for quality and performance',
        'mentions the negative value for money',
        'criticise the aesthetic and appeal',
        'mentions specific functionality issues',
        'mentions poor durability',
        'mentions the lack of confort of use',
        'mentions the bad delivery or packaging',
        'makes unfavorable comparisons',
        'warns about the product',
        'talks about unment expectations',
        'mentions frustration with performance',
        'has feelings of regret',
        'has lost trust with product or brand',
        'criticizes the brand or seller',
        'has a desire for reloution (like replacemente, refund or others)'
    ]
}

olist_dpt = DatasetPromptTemplate(dataset='olist',
                                  prompt_template=olist_prompt_template,
                                  generation_instructions=olist_key_characteristics)
