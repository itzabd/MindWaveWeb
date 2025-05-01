# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install LocalAI
RUN git clone https://github.com/mudler/LocalAI.git /LocalAI && \
    cd /LocalAI && \
    make build

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask-mail
RUN pip install --no-cache-dir matplotlib pandas
RUN pip install --no-cache-dir cloudinary
RUN pip install openai

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV LOCALAI_BASE_URL=http://localhost:8080
ENV OPENAI_API_KEY=localai

# Expose ports (Flask on 5000, LocalAI on 8080)
EXPOSE 5000 8080

# Command to run both Flask and LocalAI
CMD ["sh", "-c", "cd /LocalAI && ./local-ai & flask run --host=0.0.0.0"]