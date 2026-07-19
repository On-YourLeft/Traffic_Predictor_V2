# Use an official Python runtime as a parent image (slim version for a smaller footprint)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for PostgreSQL connectivity (psycopg2)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Create necessary directories in case they aren't pushed via git
RUN mkdir -p /app/database /app/models

# Expose port (Render sets the PORT environment variable)
EXPOSE 8000

# Start the application using a dynamic port binding 
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]