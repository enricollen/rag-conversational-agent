import argparse
from flask import Flask, request, render_template, jsonify
from langchain_community.vectorstores import Chroma
from llm import LLM
from embeddings import Embeddings

CHROMA_PATH = "chroma"

app = Flask(__name__)

# choose the embedding model
embedding_model_name = "ollama"  # "bedrock" to use BedrockEmbeddings
embeddings = Embeddings(model_name=embedding_model_name)
embedding_function = embeddings.get_embedding_function()

# initialize the llm
llm_model = LLM(model_name="llama3:8b")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query_text']
    response_text = query_rag(query_text)
    return jsonify(response=response_text)

def query_rag(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # compute similarity between embeddings of query and of pdf text chunks
    results = db.similarity_search_with_score(query_text, k=3)

    # format chunks together in context
    enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    response_text = llm_model.generate_response(context=enhanced_context_text, question=query_text)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\n\nSources: {sources}"
    return formatted_response

if __name__ == "__main__":
    app.run(debug=True)