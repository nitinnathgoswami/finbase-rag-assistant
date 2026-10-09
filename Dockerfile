FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Bake the embedding + reranker models and the vector index into the image
RUN python scripts/build_index.py && python -c "from finbase_rag.retrieval import Retriever; Retriever().reranker" || true
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
