import os

from vertexai.generative_models import GenerativeModel, GenerationConfig, ResponseValidationError, HarmBlockThreshold, \
    HarmCategory
import vertexai
from loguru import logger

from llms.base_llm import BaseLLM, AuthenticationError, HarmfulContentError, APIError


class Gemini(BaseLLM):

    def __init__(self, model_name) -> None:
        self.project_id = os.getenv('GEMINI_GCP_PROJECT_ID')
        self.location_id = os.getenv('GEMINI_GCP_LOCATION_ID')
        super().__init__(model_name=model_name)

    def authenticate(self):
        vertexai.init(project=self.project_id,
                      location=self.location_id)

    def test_authentication(self):
        try:
            self.model = GenerativeModel(self.model_name)
        except Exception as e:
            raise AuthenticationError(self.model_name)
        else:
            logger.info('Connected to Vertex AI API.')

    def generate(self, message) -> tuple[str, int, int]:
        try:
            chat = self.model.start_chat()

            response = chat.send_message(
                content=message,
                generation_config=GenerationConfig(
                    temperature=0.,
                    # top_p=0.95,
                    # #top_k=20,
                    candidate_count=1,
                    # max_output_tokens=max_tokens
                    # #stop_sequences=["\n\n\n"],
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            response_text = response.candidates[0].text

        except ResponseValidationError:
            raise HarmfulContentError(self.model_name, message)

        except Exception:
            raise APIError(self.model_name, message)

        else:
            completion_tokens = len(response_text)
            prompt_tokens = len(message)

        return response_text, completion_tokens, prompt_tokens
