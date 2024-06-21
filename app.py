import argparse
from flask import Flask, request, render_template, jsonify
from llm import LLM
from rag_retriever import RAGRetriever
from dotenv import load_dotenv
import os

load_dotenv() 

CHROMA_PATH = os.getenv('CHROMA_PATH')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')

app = Flask(__name__)

# initialize the retriever
retriever = RAGRetriever(chroma_path=CHROMA_PATH, embedding_model_name=EMBEDDING_MODEL_NAME)

# choose the llm
llm_model = LLM(model_name=LLM_MODEL_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query_text']

    # Retrieve and format results
    results = retriever.query(query_text, k=4)
    enhanced_context_text, sources = retriever.format_results(results)
    
    # Generate response from LLM
    llm_response = llm_model.generate_response(context=enhanced_context_text, question=query_text)
    response_text = f"{llm_response}<br><br> Sources: {sources}"

    return jsonify(response=response_text)

if __name__ == "__main__":
    app.run(debug=True)