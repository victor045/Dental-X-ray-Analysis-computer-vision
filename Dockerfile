# Use slim Python to keep the image small
FROM python:3.10-slim

# System deps needed by Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Avoid Python writing .pyc files & ensure unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy only requirements first (better build cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the repo
COPY . .

# optional but nice for docs
EXPOSE 7860

# IMPORTANT: bind to 7860 explicitly on HF Spaces Docker
CMD streamlit run streamlit_cloud_app.py --server.port=7860 --server.address=0.0.0.0
