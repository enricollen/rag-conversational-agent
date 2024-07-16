import os
from dotenv import load_dotenv
from llm.llm import GPTModel, OllamaModel
from llm.llm_factory import LLMFactory
from retrieval.rag_retriever import RAGRetriever

load_dotenv()

VECTOR_DB_OPENAI_PATH = os.getenv('VECTOR_DB_OPENAI_PATH')
VECTOR_DB_OLLAMA_PATH = os.getenv('VECTOR_DB_OLLAMA_PATH')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')  # 'gpt-3.5-turbo', 'GPT-4o' or local LLM like 'llama3:8b'
LLM_MODEL_TYPE = os.getenv('LLM_MODEL_TYPE')  # 'ollama', 'gpt', 'claude'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')  # 'ollama' or 'openai'
NUM_RELEVANT_DOCS = int(os.getenv('NUM_RELEVANT_DOCS'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def get_vector_db_path(embedding_model_name):
    if embedding_model_name == "openai":
        return VECTOR_DB_OPENAI_PATH
    elif embedding_model_name == "ollama":
        return VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model_name}")

def get_api_key(embedding_model_name):
    if embedding_model_name == "openai":
        return OPENAI_API_KEY
    else:
        return CLAUDE_API_KEY

# Initialize the retriever and the LLM once
vector_db_path = get_vector_db_path(EMBEDDING_MODEL_NAME)
api_key = get_api_key(EMBEDDING_MODEL_NAME)

retriever = RAGRetriever(vector_db_path=vector_db_path, embedding_model_name=EMBEDDING_MODEL_NAME, api_key=api_key)
llm_model = LLMFactory.create_llm(model_type=LLM_MODEL_TYPE, model_name=LLM_MODEL_NAME, api_key=api_key)

print(LLM_MODEL_TYPE)
print(LLM_MODEL_NAME)
print(EMBEDDING_MODEL_NAME)

def test_num_employees_alpha():
    assert query_and_validate(
        question="How many people are in the head staff inside the alpha corporation? (Answer with the number only)",
        expected_response="4",
        retriever=retriever,
        llm_model=llm_model
    )

def test_company_field_beta():
    assert query_and_validate(
        question="What is the field in which the beta enterprises operate? (Answer with few words)",
        expected_response="biotechnology and pharmaceuticals",
        retriever=retriever,
        llm_model=llm_model
    )

def test_foundation_year_gamma():
    assert query_and_validate(
        question="When was the gamma innovation society founded? (Answer with the number only)",
        expected_response="2015",
        retriever=retriever,
        llm_model=llm_model
    )

def query_and_validate(question: str, expected_response: str, retriever, llm_model):
    """
    Queries the language model (LLM) to get a response for the given question, and then validates this response
    against the expected response using the LLM itself.

    Parameters:
    question (str): The question to be asked to the LLM.
    expected_response (str): The expected response to validate against.
    retriever: An instance of the RAGRetriever used to retrieve relevant documents.
    llm_model: An instance of the LLM to generate responses.

    Returns:
    bool: True if the LLM validates that the actual response matches the expected response, False otherwise.
    """
    results = retriever.query(question, k=NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = retriever.format_results(results)
    
    # Generate response from LLM
    response_text = llm_model.generate_response(context=enhanced_context_text, question=question)

    # Use the same LLM also for response validation
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    evaluation_results_str = llm_model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(question)
    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if __name__ == "__main__":
    """
        to run tests type: 'pytest test/test_rag.py -s' in cmd
    """
    test_num_employees_alpha()
    test_company_field_beta()
    test_foundation_year_gamma()