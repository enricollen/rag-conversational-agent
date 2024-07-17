from flask import Flask, request, render_template, jsonify, redirect, url_for
from llm.llm_factory import LLMFactory
from retrieval.rag_retriever import RAGRetriever
from dotenv import load_dotenv, set_key
import os

load_dotenv()

VECTOR_DB_OPENAI_PATH = os.getenv('VECTOR_DB_OPENAI_PATH')
VECTOR_DB_OLLAMA_PATH = os.getenv('VECTOR_DB_OLLAMA_PATH')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME') # 'gpt-3.5-turbo', 'GPT-4o' or local LLM like 'llama3:8b', 'gemma2', 'mistral:7b' etc.
LLM_MODEL_TYPE = os.getenv('LLM_MODEL_TYPE')  # 'ollama', 'gpt' or 'claude'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME') # 'ollama' or 'openai'
NUM_RELEVANT_DOCS = int(os.getenv('NUM_RELEVANT_DOCS'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
ENV_PATH = '.env'

app = Flask(__name__)

# Initialize the retriever and LLM
retriever = None
llm_model = None

def get_vector_db_path(embedding_model_name):
    if embedding_model_name == "openai":
        return VECTOR_DB_OPENAI_PATH
    elif embedding_model_name == "ollama":
        return VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model_name}")

def initialize_components():
    """ Initialize the retriever and LLM components based on the current settings. """
    global retriever, llm_model
    vector_db_path = get_vector_db_path(EMBEDDING_MODEL_NAME)
    
    # Select the appropriate API key based on the embedding model
    if EMBEDDING_MODEL_NAME == "openai":
        api_key = OPENAI_API_KEY
    else:
        api_key = CLAUDE_API_KEY
    
    retriever = RAGRetriever(vector_db_path=vector_db_path, embedding_model_name=EMBEDDING_MODEL_NAME, api_key=api_key)
    llm_model = LLMFactory.create_llm(model_type=LLM_MODEL_TYPE, model_name=LLM_MODEL_NAME, api_key=api_key)
    print(f"Instantiating model type: {LLM_MODEL_TYPE} | model name: {LLM_MODEL_NAME} | embedding model: {EMBEDDING_MODEL_NAME}")

initialize_components()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html', 
                           llm_model_name=LLM_MODEL_NAME,
                           llm_model_type=LLM_MODEL_TYPE,
                           embedding_model_name=EMBEDDING_MODEL_NAME,
                           num_relevant_docs=NUM_RELEVANT_DOCS,
                           openai_api_key=OPENAI_API_KEY)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global LLM_MODEL_NAME, LLM_MODEL_TYPE, EMBEDDING_MODEL_NAME, NUM_RELEVANT_DOCS, OPENAI_API_KEY
    LLM_MODEL_NAME = request.form['llm_model_name']
    LLM_MODEL_TYPE = request.form['llm_model_type']
    EMBEDDING_MODEL_NAME = request.form['embedding_model_name']
    NUM_RELEVANT_DOCS = int(request.form['num_relevant_docs'])
    OPENAI_API_KEY = request.form['openai_api_key']

    # Update the .env file
    set_key(ENV_PATH, 'LLM_MODEL_NAME', LLM_MODEL_NAME)
    set_key(ENV_PATH, 'LLM_MODEL_TYPE', LLM_MODEL_TYPE)
    set_key(ENV_PATH, 'EMBEDDING_MODEL_NAME', EMBEDDING_MODEL_NAME)
    set_key(ENV_PATH, 'NUM_RELEVANT_DOCS', str(NUM_RELEVANT_DOCS))
    set_key(ENV_PATH, 'OPENAI_API_KEY', OPENAI_API_KEY)
    
    # Reinitialize the components (llm and retriever objects)
    initialize_components()
    print(f"Updating model type: {LLM_MODEL_TYPE} | model name: {LLM_MODEL_NAME} | embedding model: {EMBEDDING_MODEL_NAME}")
    return redirect(url_for('admin'))

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query_text']
    # Retrieve and format results
    results = retriever.query(query_text, k=NUM_RELEVANT_DOCS)
    enhanced_context_text, sources = retriever.format_results(results)
    # Generate response from LLM
    llm_response = llm_model.generate_response(context=enhanced_context_text, question=query_text)
    sources_html = "<br>".join(sources)
    response_text = f"{llm_response}<br><br>Sources:<br>{sources_html}<br><br>Response given by: {LLM_MODEL_NAME}"
    return jsonify(response=response_text)

if __name__ == "__main__":
    app.run(debug=True)