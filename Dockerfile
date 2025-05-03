FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and LocalAI binary
COPY . .

# Ensure LocalAI binary is executable
RUN chmod +x /app/local-ai

# Expose ports: Flask (5000) + LocalAI (8080)
EXPOSE 5000 8080

# Healthcheck for LocalAI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ready || exit 1

# Start both services
CMD ["sh", "-c", "./local-ai --models-path /models & flask run --host=0.0.0.0"]
