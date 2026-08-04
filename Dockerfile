FROM python:3.12-slim

# Build tools for the C++ extension
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Build the C++ extension inside the container
RUN cd cpp && make

# Adjust if your Makefile doesn't already copy the .so into orchestrator/
# RUN cp cpp/build/minibrain_cpp*.so python/orchestrator/

EXPOSE 10000
CMD ["python", "python/orchestrator/main.py"]
