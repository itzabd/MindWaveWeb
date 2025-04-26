FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to non-root user
RUN useradd -m appuser && mkdir /app && chown appuser:appuser /app
USER appuser
WORKDIR /app

# Install Python dependencies directly (no virtual env)
RUN pip install --no-cache-dir \
    setuptools==68.2.2 \
    pip==23.3.2 \
    gunicorn==21.2.0 \
    aiohttp==3.9.0

# Install main requirements
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Supabase stack explicitly
RUN pip install --no-cache-dir \
    supabase==2.7.0 \
    gotrue==2.12.0 \
    postgrest==0.16.11 \
    realtime==2.4.2

# Install EEG-LLM requirements
RUN pip install --no-cache-dir \
    pandas==1.5.3 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    mne==1.6.1 \
    backoff==2.2.1 \
    scikit-learn==1.3.2 \
    openai==1.30.2 \
    matplotlib==3.7.5

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 5000

# Use direct gunicorn path (no virtual env)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]