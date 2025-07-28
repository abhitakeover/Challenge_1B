FROM --platform=linux/amd64 python:3.9-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    poppler-utils \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for NLTK data and download required resources
RUN mkdir -p /usr/share/nltk_data && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data')" && \
    python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data')"

# Set environment variable for NLTK to use our data directory
ENV NLTK_DATA=/usr/share/nltk_data

# Copy application code
COPY main.py .

# Entry point
ENTRYPOINT ["python", "main.py"]