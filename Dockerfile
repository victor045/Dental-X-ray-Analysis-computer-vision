FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV DATA_ROOT=/app/data
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "correct_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]