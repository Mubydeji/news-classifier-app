FROM python:3.11-slim

WORKDIR /code

# CPU-only torch, matches the pneumonia app's approach: Render's free tier
# is CPU-only, no reason to pull in unused CUDA libraries.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
