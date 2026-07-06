# Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# models/ буде скопійована якщо є в репо

EXPOSE 8000
CMD ["python", "5_serve.py"]