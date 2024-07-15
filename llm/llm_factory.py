from llm.llm import LLM, GPTModel, OllamaModel, AnthropicModel

class LLMFactory:
    @staticmethod
    def create_llm(model_type: str, model_name: str, api_key: str = None) -> LLM:
        if model_type == 'ollama':
            return OllamaModel(model_name)
        elif model_type == 'gpt':
            return GPTModel(model_name, api_key)
        elif model_type == 'claude':
            return AnthropicModel(model_name, api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")