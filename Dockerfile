### Dockerfile for VisionDoc RAG - Final Production Version ###

# 1. Base Image: Use a recent, patched version of Python on Debian.
FROM python:3.11-slim-bookworm

# 2. System Dependencies: Install Poppler and Tesseract as root.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory
WORKDIR /app

# 4. Install Python Dependencies as root
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all necessary application code
# We only copy the 'server' directory, which is our backend application.
COPY ./server /app

# 6. Expose Port
EXPOSE 8000

# 7. Run Command: Start the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
