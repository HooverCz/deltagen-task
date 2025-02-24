init:
    brew install poppler tesseract libmagic
    pip install uv
    uv venv
    uv sync
    source .venv/bin/activate

run:
    python run_ingest_vector_db.py
    streamlit run app.py
