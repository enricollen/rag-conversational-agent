from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Basing only on the following context:

{context}

---

Answer the following question: {question}
"""

class LLM:
    def __init__(self, model_name: str):
        self.model = Ollama(model=model_name)

    def generate_response(self, context: str, question: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)
        response_text = self.model.invoke(prompt)
        return response_text