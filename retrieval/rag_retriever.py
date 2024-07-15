from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embeddings.embeddings import Embeddings

class RAGRetriever:
    def __init__(self, vector_db_path: str, embedding_model_name: str, api_key: str):
        self.vector_db_path = vector_db_path
        embeddings = Embeddings(model_name=embedding_model_name, api_key=api_key)
        self.embedding_function = embeddings.get_embedding_function()
        self.db = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embedding_function)

    def query(self, query_text: str, k: int = 4):
        # compute similarity between embeddings of query and of pdf text chunks
        results = self.db.similarity_search_with_score(query_text, k=k)
        return results

    def format_results(self, results: list[tuple[Document, float]]):
        enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = set(self.format_source(doc.metadata) for doc, _score in results)  # set to ensure uniqueness
        return enhanced_context_text, list(sources)

    def format_source(self, metadata: dict):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        filename = source.split("\\")[-1]  # extract filename
        return f"{filename} page {page}"