FROM python:3.11-slim  # Using 3.11 for better compatibility

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional EEG processing dependencies
RUN pip install --no-cache-dir \
    matplotlib \
    pandas \
    scipy \
    scikit-learn \
    pygdf  # For GDF file support

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV PYTHONPATH=/app

# Create directory for EEG data
RUN mkdir -p /app/eeg_data

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]