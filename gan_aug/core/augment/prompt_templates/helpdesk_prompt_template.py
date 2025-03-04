from ..dataset_prompt_template import DatasetPromptTemplate


helpdesk_prompt_template = """INSTRUCTION
Here are some examples of helpdesk emails texts labeled as one of: "General Inquiry", "Human Resources", "Billing and Payments", "Sales and Pre-Sales", "IT Support", "Customer Service", "Product Support", "Returns and Exchanges", "Service Outages and Maintenance", "Technical Support".
Generate new texts for these categories, keeping the tone and style similar to the examples.

OUTPUT
Generate {num} new text samples.
Your output should be in JSON format and you shoud onyl return this JSON.
Return a list of objects with the keys `text` (for the generated text) and
`label` for its label

EXAMPLES
{examples_text}
"""

helpdesk_dpt = DatasetPromptTemplate(dataset='helpdesk', prompt_template=helpdesk_prompt_template)
