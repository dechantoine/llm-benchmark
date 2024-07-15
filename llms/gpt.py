import os

import openai
from loguru import logger

from llms.base_llm import BaseLLM, AuthenticationError, HarmfulContentError, APIError


class GPT(BaseLLM):

    def __init__(self, model_name) -> None:
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.api_endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION')

        super().__init__(model_name=model_name)

    def authenticate(self):
        self.client = openai.AzureOpenAI(api_key=self.api_key,
                                         api_version=self.api_version,
                                         azure_endpoint=self.api_endpoint)

    def test_authentication(self):
        try:
            self.client.models.list()
        except Exception:
            raise AuthenticationError(self.model_name)
        else:
            logger.info('Connected to Azure OpenAI API.')

    def format_message_classification(self, system_prompt: str, user_message: str, example_input: str,
                                      example_output: str) -> list[dict[str, str]]:
        system_prompt_content = f"""{system_prompt}
        Here is an example of the input and output:
        Message: '{example_input}'
        Label: {example_output} """

        message = [{'role': 'system',
                    'content': system_prompt_content,
                    },
                   {'role': 'user',
                    'content': user_message,
                    }]
        return message

    def generate(self, message) -> tuple[str, int, int]:
        if type(message) is str:
            message = [{'role': 'user', 'content': message}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
            )
            response_text = response.choices[0].message.content
        except openai.BadRequestError as e:
            raise APIError(self.model_name, message) from e
        else:
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens

        if response.choices[0].finish_reason == 'content_filter':
            raise HarmfulContentError(self.model_name, message)

        return response_text, completion_tokens, prompt_tokens
