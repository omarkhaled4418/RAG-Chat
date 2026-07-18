FROM python:3.10-slim

# Create a user with uid 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user

WORKDIR /app

# Install necessary system dependencies for PyMuPDF and FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories and give ownership to the new user
RUN mkdir -p /app/data /app/uploads && chown -R user:user /app

USER user

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "run:application"]
