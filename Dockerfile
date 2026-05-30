FROM python:3.12-slim-bookworm

# ENV PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal OS deps, update packages to pick up security fixes, then clean up
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt rquirements.txt

# Upgrade packaging tools then install Python deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . . 

# # Create a non-root user for runtime
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

EXPOSE 8000

# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]