from abc import ABC
from enum import Enum

from pydantic import BaseModel, ValidationError
from loguru import logger
from retry.api import retry_call

from llms.mixin import format_preprompt


class HarmfulContentError(Exception):
    """Harmful content error.
    Raised when the response is blocked for harmful content.

    Args:
        model (str): model name.
        prompt (str): prompt passed to the API.
    """

    def __init__(self, model: str, prompt: str):
        self.prompt = prompt
        self.model = model
        super().__init__(
            f'Response blocked for harmful content by API for model {self.model} with the prompt: {self.prompt}')


class AuthenticationError(Exception):
    """Authentication error.
    Raised when the API returns an authentication error.

    Args:
        model (str): model name.
    """

    def __init__(self, model: str):
        self.model = model
        super().__init__(f'Authentication error for model {self.model}')


class APIError(Exception):
    """API error.
    Raised when the API returns an error.

    Args:
        model (str): model name.
        prompt (str): prompt passed to the API.
    """

    def __init__(self, model: str, prompt: str):
        self.prompt = prompt
        self.model = model
        super().__init__(f'API error for model {self.model} with the prompt: {self.prompt}')


class BaseLLM(ABC):

    @logger.catch
    def __init__(self, model_name: str, tries: int = 3, delay: int = 30):
        self.model_name = model_name
        self.tries = tries
        self.delay = delay

        self.authenticate()
        self.test_authentication()

    @logger.catch
    def authenticate(self):
        """Authenticate."""
        pass

    @logger.catch(exclude=AuthenticationError)
    def test_authentication(self):
        """Test authentication."""
        pass

    @logger.catch
    def tokenize(self, text: str) -> list[int]:
        pass

    @logger.catch
    def token_eos(self) -> int:
        pass

    @logger.catch
    def n_vocab(self) -> int:
        pass

    @logger.catch(exclude=(HarmfulContentError, APIError))
    def generate(self, message) -> tuple[str, int, int]:
        """Generate predictions for a single row.

        Args:
            message (str): message

        Returns:
            response (str): response
            completion_tokens (int): completion tokens
            prompt_tokens (int): prompt tokens
        """
        pass

    @logger.catch
    def format_message_classification(self, system_prompt: str, user_message: str, example_input: str,
                                      example_output: str) -> str:
        """Format message for classification tasks.

        Args:
            system_prompt (str): system prompt
            user_message (str): user message
            example_input (str): example input
            example_output (str): example output

        Returns:
            prompt (str): formatted prompt
        """
        prompt = f"""{system_prompt}

                    Here is an example of the input and output:
                    Message: '{example_input}'
                    Label: {example_output}

                    Now, please provide your response:
                    Message: '{user_message}'
                    Label: """

        return prompt

    @logger.catch
    def classify(self,
                 preprompt: str,
                 prompt: str,
                 labels: list[str],
                 predict_labels_index: bool,
                 example_input: str,
                 example_output: str,
                 validation_error_label: str = "VALIDATION_ERROR",
                 response_blocked_error_label: str = "RESPONSE_BLOCKED_ERROR",
                 harmful_content_error_label: str = "HARMFUL_CONTENT_ERROR"
                 ) \
            -> tuple[str, int, int]:
        """Classify a single row.

        Args:
            model (str): model name
            preprompt (str): preprompt
            prompt (str): prompt
            labels (list[str]): list of possible labels
            predict_labels_index (bool): whether the model must predict labels index (less costly)
            example_input (str): example input
            example_output (str): example output
            validation_error_label (str): validation error label
            response_blocked_error_label (str): response blocked error label
            harmful_content_error_label (str): harmful content error label

        Returns:
            label (str): predicted label
            completion_tokens (int): completion tokens
            prompt_tokens (int): prompt tokens
        """

        class Label(BaseModel):
            label: Enum("Labels", {l: l for l in labels})

        message = self.format_message_classification(
            system_prompt=format_preprompt(preprompt, labels),
            user_message=prompt,
            example_input=example_input,
            example_output=example_output
        )

        try:
            response, completion_tokens, prompt_tokens = retry_call(f=self.generate,
                                                                    exceptions=(APIError,),
                                                                    fargs=[message],
                                                                    tries=self.tries,
                                                                    delay=self.delay)

        except HarmfulContentError as e:
            logger.warning(f'{e}')
            label = harmful_content_error_label
            completion_tokens = 0
            prompt_tokens = 0

        except APIError as e:
            logger.warning(f'{e}')
            label = response_blocked_error_label
            completion_tokens = 0
            prompt_tokens = 0

        else:
            try:
                response_clean = "".join([label if response.find(label) > -1 else "" for label in labels])
                label = Label(label=response_clean).label.value
            except ValidationError:
                logger.warning(f'Validation error from model {self.model_name}: {response} is not a valid label.')
                label = validation_error_label

        return label, completion_tokens, prompt_tokens
