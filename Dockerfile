# Face Attendance System — Hugging Face Spaces (Docker SDK)
# Reference: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    DATA_DIR=/data

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user - required for HF Spaces Dev Mode and good practice generally
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Switch to user
USER user

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt requirements.txt

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Copy the rest of the application
COPY --chown=user . /app

# Make startup script executable if it exists
RUN if [ -f /app/startup.sh ] ; then chmod +x /app/startup.sh ; fi

# Create data directory for persistent storage
RUN mkdir -p /data

# Expose the port
EXPOSE 7860

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/healthz || exit 1

# Run the application with gunicorn directly (without startup script to avoid potential issues)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "--preload"]