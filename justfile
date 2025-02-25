init:
    brew install poppler tesseract libmagic
    pip install uv
    uv venv
    uv sync
    source .venv/bin/activate

run:
    rm -rf vector_db
    python 01_ingest_vector_db.py
    streamlit run 02_app.py
