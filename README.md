# Project Title

This project ingests documents and serves a Streamlit web application that interacts with a vector database. It provides tools to ingest documents into the vector database and then explore them through a responsive web interface.

## Features

- Ingest documents into a vector database.
- Serve a Streamlit application for interactive exploration.
- Built with Python and uv.

## Setup

### Prerequisites

- macOS with Homebrew installed.
- Python (version as specified in .python-version).
- pip package manager.

### Installation

Run the following command to install dependencies and set up the Python virtual environment:

    just init

Internally, the "just init" command performs the following actions:

    brew install poppler tesseract libmagic
    pip install uv
    uv venv
    uv sync
    source .venv/bin/activate

### LLM API Keys (environment variables)

Make sure to populate values for `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`  in the `.env` file with valid values.
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` will be provided separately (e.g. via e-mail).

### Running the Application

After initialization, start the application by running:

    just run

This command will:
- Execute the ingestion script (run_ingest_vector_db.py) to load documents into the vector database.
- Launch the Streamlit application (app.py).

Internally, the "just run" command performs the following actions:

    python run_ingest_vector_db.py
    streamlit run app.py


## Additional Information

If needed, activate the virtual environment manually:

    source .venv/bin/activate

For any issues or further information, please refer to the documentation or contact the project maintainer.


## Architecture and Design Decisions
#### Document Parsing Library → unstructured
- The unstructured library is easy to use and provides a comprehensive toolkit for parsing text, images, and tables from PDFs, along with many other document formats.
- It offers out-of-the-box chunking, including effective strategies such as title-based chunking, which helps maintain contextual integrity.

#### Vector Store → ChromaDB
- ChromaDB is lightweight, easy to use locally, and supports Multi-Vector Retrieval.
- The Multi-Vector Retriever enhances retrieval performance by embedding and indexing diverse data types beyond plain text, including tables and images.
- This approach allows structured data, such as table summaries or image descriptions, to be embedded and retrieved efficiently in RAG (Retrieval-Augmented Generation) pipelines.
- When a user query matches a summary, the full document (or image/table) is retrieved and passed to an LLM for a more detailed response, ensuring no critical context is lost.

#### LLMs → Azure OpenAI (Embeddings + GPT-4o-mini)
- Many PDFs related to company information include images, often containing crucial data in the form of charts or diagrams.
- This necessitates a multimodal language model capable of processing images by generating precise descriptions.
- Due to limited time constraints, I chose Azure OpenAI-hosted models because:
    - I have extensive experience using them.
    - They are quick to set up and integrate.
    - They provide reliable performance for both text and multimodal processing.

#### Assistant Architecture → LangGraph
- LangGraph enables flexible control over the workflow and execution logic of an LLM-based application.
- It allows for defining complex flows beyond simple linear chains.
- With state management, complex data structures can be efficiently passed between nodes without overloading the model's context window with excessive tokens.
- Debugging is convenient using LangSmith, making it easier to monitor and optimize execution paths.
