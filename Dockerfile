### Dockerfile for VisionDoc RAG - Production Grade ###

# Use the official, high-performance image for FastAPI applications.
# This image is maintained by the creator of FastAPI and includes Uvicorn and Gunicorn.
# It's built on a secure base and is optimized for production workloads.
FROM tiangolo/uvicorn-gunicorn:python3.11

# The base image already has a non-root user and best practices configured.

# Install system dependencies required for OCR and PDF processing.
# We run this as the root user temporarily and then switch back.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
USER uvicorn

# The base image sets the WORKDIR to /app, so we don't need to.

# Copy and install Python dependencies.
# This is done before copying the app code to leverage Docker's layer caching.
COPY ./server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire server application code into the container.
# The base image is configured to look for the app in the /app directory.
COPY ./server /app

# The base image already exposes port 80 and has a CMD to start the server.
# We don't need to specify EXPOSE or CMD.
# The server will look for an `app` object in a `main.py` file inside the `/app` directory by default.