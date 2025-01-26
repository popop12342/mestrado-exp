import random
from pydantic import BaseModel, Field


class DatasetPromptTemplate(BaseModel):
    dataset: str
    prompt_template: str
    generation_instructions: dict[str, list[str]] = Field(default_factory=dict,
                                                          description='Generation instructions for the LLM. The key of\
                                                            this dictionary is a label (eg ``positive``) and the value\
                                                                is a list of characteristics to instruct the LLM\
                                                                    (eg ``comments about the good acting``)')

    def build_template(self) -> str:
        """Builds a prompt template for a dataset. If has specific generation
        instruction apply those by random selection policy and characteristics.
        Otherwise return default prompt template.

        Returns:
            str: prompt template
        """
        if not self.generation_instructions:
            return self.prompt_template
        polarities = list(self.generation_instructions.keys())
        polarity = random.choice(polarities)
        characteristic = random.choice(self.generation_instructions[polarity])
        specific_instruction = f'Generate a {polarity} review that {characteristic}.'
        return self.prompt_template.format(specific_instruction)
