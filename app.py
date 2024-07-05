import argparse
from flask import Flask, request, render_template, jsonify
from llm.llm_factory import LLMFactory
from retrieval.rag_retriever import RAGRetriever
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME') # 'gpt-3.5-turbo', 'GPT-4o' or local LLM like 'llama3:8b'
LLM_MODEL_TYPE = os.getenv('LLM_MODEL_TYPE')  # 'ollama' or 'gpt'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME') # 'ollama' or 'openai'
NUM_RELEVANT_DOCS = int(os.getenv('NUM_RELEVANT_DOCS'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Initialize the retriever
retriever = RAGRetriever(chroma_path=CHROMA_PATH, embedding_model_name=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)

# Choose the LLM
llm_model = LLMFactory.create_llm(model_type=LLM_MODEL_TYPE, model_name=LLM_MODEL_NAME, api_key=OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query_text']

    # Retrieve and format results
    results = retriever.query(query_text, k=NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = retriever.format_results(results)
    
    # Generate response from LLM
    llm_response = llm_model.generate_response(context=enhanced_context_text, question=query_text)
    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}"

    return jsonify(response=response_text)

if __name__ == "__main__":
    app.run(debug=True)