# Base image
FROM python:3.10-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Source code
COPY . .

# Dependencies
RUN pip install -r requirements.txt

# Configuration
EXPOSE 8192
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]
