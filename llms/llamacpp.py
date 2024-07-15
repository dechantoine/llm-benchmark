import os

from llama_cpp import Llama

from huggingface_hub import hf_hub_download

from llms.base_llm import BaseLLM


class LlamaCPP(BaseLLM):

    def __init__(self, model_name: str, prompt_template) -> None:
        super().__init__(model_name)
        self.prompt_template = prompt_template

        model_paths = model_name.split("/")

        if not os.path.exists("temp"):
            os.makedirs("temp")

        if not os.path.exists(os.path.join("temp", model_paths[2])):
            hf_hub_download(repo_id=model_paths[0] + "/" + model_paths[1],
                            filename=model_paths[2],
                            repo_type="model",
                            local_dir="temp",
                            local_dir_use_symlinks=False)

        self.model = Llama(
            model_path=os.path.join("temp", model_paths[2]),
            n_ctx=2048,
            # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=7,
            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=-1,
            # The number of layers to offload to GPU, if you have GPU acceleration available
            # Set to 0 if no GPU acceleration is available on your system.
        )

    def tokenize(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode("utf-8"))

    def token_eos(self) -> int:
        return self.model.token_eos()

    def n_vocab(self) -> int:
        return self.model.n_vocab()

    def format_message_classification(self, system_prompt: str, user_message: str, example_input: str,
                                      example_output: str) -> list[dict[str, str]]:
        message = self.prompt_template.format(system_prompt=system_prompt,
                                              example_input=example_input,
                                              example_output=example_output,
                                              user_message=user_message)

        return message


    def generate(self, message) -> tuple[str, int, int]:
        if type(message) is str:
            message = f"<s>[INST] {message} [/INST]"

        #max_tokens = len(str(len(labels)))+1 if predict_labels_index else max([len(label) for label in labels])/2

        response = self.model(message,
                              temperature=0.0,
                              #max_tokens=max_tokens,
                              stop=["</s>"],
                              #logit_bias=logit_bias,
                              echo=False)

        response_text = response["choices"][0]["text"]
        completion_tokens = response["usage"]["completion_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]

        return response_text, completion_tokens, prompt_tokens
