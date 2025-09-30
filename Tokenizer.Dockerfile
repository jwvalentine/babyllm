FROM python:3.11

WORKDIR /app

# Install FastAPI + Uvicorn + Hugging Face Tokenizers
RUN pip install --no-cache-dir fastapi uvicorn[standard] tokenizers

# Copy the tokenizer service script
COPY tokenizer_server.py .

EXPOSE 8082
CMD ["uvicorn", "tokenizer_server:app", "--host", "0.0.0.0", "--port", "8082"]
