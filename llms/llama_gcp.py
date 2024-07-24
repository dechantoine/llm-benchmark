import subprocess
import requests
import json
import os

from loguru import logger

from llms.base_llm import BaseLLM, AuthenticationError, HarmfulContentError, APIError


class LlamaGCP(BaseLLM):

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.location = os.getenv("LLAMA_GCP_LOCATION")
        self.project_id = os.getenv("LLAMA_GCP_PROJECT_ID")
        self.endpoint = f"https://{self.location}-aiplatform.googleapis.com"
        self.url_endpoint = (f"{self.endpoint}/v1beta1/projects/{self.project_id}/locations/{self.location}"
                             f"/endpoints/openapi/chat/completions")

        super().__init__(model_name=model_name)

    def authenticate(self):
        subprocess.run("gcloud auth login", shell=True)
        process = subprocess.Popen("gcloud auth print-access-token", stdout=subprocess.PIPE, shell=True)
        (access_token_bytes, err) = process.communicate()
        self.access_token = access_token_bytes.decode('utf-8').strip()

    def test_authentication(self):
        url_list = f"{self.endpoint}/v1beta1/projects/{self.project_id}/locations/{self.location}/publishers/meta/models"
        try:
            requests.get(url_list)
        except Exception as e:
            raise AuthenticationError(self.model_name)
        else:
            logger.info('Connected to Llama GCP API.')

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

        data = {
            "model": f"meta/{self.model_name}",
            "messages": message,
            "temperature": 0.0,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.url_endpoint, headers=headers, json=data)

        logger.info(response)

        try:
            response_dict = response.json()
            response_text = response_dict['choices'][0]['message']['content']

        except json.JSONDecodeError as e:
            logger.warning("Error decoding JSON:", e)
            logger.warning("Raw response:", response)  # Print raw response if parsing fails
            raise APIError(self.model_name, message) from e

        else:
            completion_tokens = response_dict["usage"]["completion_tokens"]
            prompt_tokens = response_dict["usage"]["prompt_tokens"]

        return response_text, completion_tokens, prompt_tokens
