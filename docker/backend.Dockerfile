FROM python:3.11-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_REQUIRE_VIRTUALENV=0
RUN apt-get update && apt-get install -y ffmpeg exiftool tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade "pip>=25.3,<27" \
    && python -m pip install --no-cache-dir --no-require-hashes -r requirements.txt
COPY . .
CMD ["uvicorn","backend.app.main:app","--host","0.0.0.0","--port","8000"]
