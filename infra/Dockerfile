FROM python:3.11-slim

# Make Python quieter/faster in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OS deps: ffmpeg for video, curl for healthcheck, certs for HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your service code
COPY service /app/service

# Non-root user for safety + give it ownership of /app
RUN useradd -u 10001 -m appuser \
 && mkdir -p /app/files \
 && chown -R appuser:appuser /app
USER appuser

# Keep your current import style working
ENV PYTHONPATH=/app/service

EXPOSE 8080

# Healthcheck hits your FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Start the API
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
