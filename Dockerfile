# Hexagonal Geometric Test Engine - one container running the Streamlit app on port 8501.
FROM python:3.12-slim

# Runtime libraries opencv-python-headless needs on Debian slim (no GUI libraries: headless build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEX_DATA_DIR=/app/data \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install dependencies first so code edits don't force a full reinstall on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 hex \
    && mkdir -p /app/data \
    && chown -R hex:hex /app

COPY --chown=hex:hex hexengine/ ./hexengine/
COPY --chown=hex:hex app/ ./app/
COPY --chown=hex:hex samples/ ./samples/
COPY --chown=hex:hex tests/ ./tests/
COPY --chown=hex:hex .streamlit/ ./.streamlit/

USER hex

EXPOSE 8501

# curl isn't in the slim image, so use Python's standard library for the health probe.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
