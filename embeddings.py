from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

class Embeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_embedding_function(self):
        if self.model_name == "bedrock":
            return BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
        elif self.model_name == "ollama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")