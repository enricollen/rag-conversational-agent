from openai import OpenAI

class OpenAIEmbeddings:
    """
    class that implements two methods to be called from Chroma
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str):
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding