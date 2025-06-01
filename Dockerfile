# Base image
FROM python:3.10-bookworm
RUN apt-get update && apt-get install dumb-init
RUN update-ca-certificates

# Source code
COPY . .

# Dependencies
RUN pip install -r requirements.txt

# Configuration
EXPOSE 8192
CMD ["dumb-init", "--", "fastapi", "run", "app.py", "--port", "8192"]