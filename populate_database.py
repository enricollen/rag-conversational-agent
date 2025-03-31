import argparse
import os
import shutil
from embeddings.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATA_PATH = os.getenv('DATA_PATH')
VECTOR_DB_OPENAI_PATH = os.getenv('VECTOR_DB_OPENAI_PATH')
VECTOR_DB_OLLAMA_PATH = os.getenv('VECTOR_DB_OLLAMA_PATH')

def main():
    # check whether the database should be cleared or not (using the --clear flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", nargs="?", const="both", choices=["ollama", "openai", "both"], help="Reset the database.")
    parser.add_argument("--embedding-model", type=str, default="openai", help="The embedding model to use (ollama or openai).")
    args = parser.parse_args()

    if args.reset:
        reset_databases(args.reset)
        return

    # choose the embedding model
    embeddings = Embeddings(model_name=args.embedding_model, api_key=OPENAI_API_KEY)
    embedding_function = embeddings.get_embedding_function()

    # determine the correct path for the database based on the embedding model
    if args.embedding_model == "openai":
        db_path = VECTOR_DB_OPENAI_PATH
    elif args.embedding_model == "ollama":
        db_path = VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError("Unsupported embedding model specified.")

    # load the existing database
    db = Chroma(
        persist_directory=db_path, embedding_function=embedding_function
    )

    # create (or update) the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, db)

def reset_databases(reset_choice):
    if reset_choice in ["openai", "both"]:
        if ask_to_clear_database("openai"):
            print("✨ Rebuilding OpenAI Database")
            clear_database("openai")
            rebuild_database("openai")

    if reset_choice in ["ollama", "both"]:
        if ask_to_clear_database("ollama"):
            print("✨ Rebuilding Ollama Database")
            clear_database("ollama")
            rebuild_database("ollama")

def ask_to_clear_database(embedding_model):
    response = input(f"Do you want to override the existing {embedding_model} database? (yes/no): ").strip().lower()
    return response == 'yes'

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], db):
    # calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"➕ Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # create IDs like "data/alpha_society.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if the page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # calculate the unique chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # add it to the page meta-data
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(embedding_model):
    if embedding_model == "openai":
        db_path = VECTOR_DB_OPENAI_PATH
    elif embedding_model == "ollama":
        db_path = VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError("Unsupported embedding model specified.")

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

def rebuild_database(embedding_model):
    if embedding_model == "openai":
        embeddings = Embeddings(model_name="openai", api_key=OPENAI_API_KEY)
        db_path = VECTOR_DB_OPENAI_PATH
    elif embedding_model == "ollama":
        embeddings = Embeddings(model_name="ollama", api_key=OPENAI_API_KEY)
        db_path = VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError("Unsupported embedding model specified.")

    embedding_function = embeddings.get_embedding_function()

    # load the existing database
    db = Chroma(
        persist_directory=db_path, embedding_function=embedding_function
    )

    # create (or update) the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, db)

if __name__ == "__main__":
    main()