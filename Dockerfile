FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p python/orchestrator/static && cp index.html python/orchestrator/static/index.html

RUN cd cpp && make

WORKDIR /app/python
EXPOSE 10000
CMD ["python", "-m", "orchestrator.main"]
