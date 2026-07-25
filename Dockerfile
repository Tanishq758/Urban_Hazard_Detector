FROM python:3.11-slim

# opencv-python-headless still occasionally needs these shared libs on a
# minimal Debian base even though it's the "headless" build — this avoids
# the classic "ImportError: libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces (Docker SDK) runs the container as a non-root user by
# convention — create one and do everything below as that user.
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=appuser:appuser backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --user -r backend/requirements.txt

COPY --chown=appuser:appuser backend/ backend/
COPY --chown=appuser:appuser frontend/ frontend/

# HF Spaces routes external HTTPS traffic to port 7860 by default — this
# must match `app_port` in README.md's Space metadata block.
EXPOSE 7860

# ADMIN_PASSWORD should be set as a Space "secret" (Settings > Variables and
# secrets), not baked into the image or committed here.
# DB_PATH is left unset on purpose -> falls back to the OS temp dir (see
# database.py), which means reports reset whenever the Space restarts/sleeps.
# Fine for a demo; add HF's paid persistent storage + DB_PATH=/data/reports.db
# if you need reports to survive restarts.

WORKDIR $HOME/app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
