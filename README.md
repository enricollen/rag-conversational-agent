A simple local Retrieval-Augmented Generation (RAG) chatbot that can answer to questions by acquiring information from personal pdf documents.

(please, if you find this content useful please consider leaving a star ⭐)

##  What is Retrieval-Augmented Generation (RAG)?
<div style="text-align: center;">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*J7vyY3EjY46AlduMvr9FbQ.png" alt="rag_pipeline" width="600" height="300">
</div>
Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of information retrieval and natural language generation. In a RAG system, a retriever fetches relevant documents or text chunks from a database, and then a generator produces a response based on the retrieved context.

1. **Data Indexing**
- Documents: This is the starting point where multiple documents are stored.
- Vector DB: The documents are processed and indexed into a Vector Database.

2. **User Query**
- A user query is input into the system, which interacts with the Vector Database.

3. **Data Retrieval & Generation**
- Top-K Chunks: The Vector Database retrieves the top-K relevant chunks based on the user query.
- LLM (Large Language Model): These chunks are then fed into a Large Language Model.
- Response: The LLM generates a response based on the relevant chunks.

## 🏗️ Implementation Components
For this project, i exploited the following components to build the RAG architecture:
1. **Chroma**: A vector database used to store and retrieve document embeddings efficiently.
2. **Flask**: Framework for rendering web page and handling user interactions.
3. **Ollama**: Manages the local language model for generating responses.
4. **LangChain**: A framework for integrating language models and retrieval systems.

## 🛠️ Setup and Local Deployment

1. **Choose Your Setup**:
   - You have three different options for setting up the LLMs:
     1. Local setup using Ollama.
     2. Using the OpenAI API for GPT models.
     3. Using the Anthropic API for Claude models.

### Option 1: Local Setup with Ollama

- **Download and install Ollama on your PC**:
   - Visit [Ollama's official website](https://ollama.com/download) to download and install Ollama. Ensure you have sufficient hardware resources to run the local language model.
   - Pull a LMM of your choice:
   ```sh
   ollama pull <model_name>  # e.g. ollama pull llama3:8b

### Option 2: Use OpenAI API for GPT Models
- **Set up OpenAI API**: you can sign up and get your API key from [OpenAI's website](https://openai.com/api/).

### Option 3: Use Anthropic API for Claude Models
- **Set up Anthropic API**: you can sign up and get your API key from [Anthropic's website](https://www.anthropic.com/api).

## Common Steps

2. **Clone the repository and navigate to the project directory**:
    ```sh
    git clone https://github.com/enricollen/rag-conversational-agent.git
    cd rag-conversational-agent
    ```

3. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Insert you own PDFs in /data folder**

6. **Run once the populate_database script to index the pdf files into the vector db:**
    ```sh
    python populate_database.py
    ```

7. **Run the application:**
    ```sh
    python app.py
    ```

8. **Navigate to `http://localhost:5000/`**

9. **If needed, click on ⚙️ icon to access the admin panel and adjust app parameters**

10. **Perform a query** 

## 🚀 Future Improvements
Here are some ideas for future improvements:
- [x] Add OpenAI LLM GPT models compatibility (3.5 turbo, 4, 4-o)
- [x] Add Anthropic Claude LLM models compatibility (Claude 3.5 Sonnet, Claude 3 Sonnet, Claude 3 Opus, Claude 3 Haiku)
- [x] Add unit testing to validate the responses given by the LLM
- [x] Add an admin user interface in web UI to choose interactively the parameters like LLMs, embedding models etc.
- [ ] Add Langchain Tools compatibility, allowing users to define custom Python functions that can be utilized by the LLMs.
- [ ] Add web scraping in case none of the personal documents contain relevant info w.r.t. the query

## 📹 Demo Video
Watch the demo video below to see the RAG Chatbot in action:

[![YT Video](https://img.youtube.com/vi/_JVt5gwwZq0/0.jpg)](https://www.youtube.com/watch?v=_JVt5gwwZq0)

The demo was run on my PC with the following specifications:
- **Processor**: Intel(R) Core(TM) i7-14700K 3.40 GHz
- **RAM**: 32.0 GB
- **GPU**: NVIDIA GeForce RTX 3090 FE 24 GB
