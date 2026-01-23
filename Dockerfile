# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project (including src and templates)
COPY . .

# 6. Expose the port Flask runs on
EXPOSE 8080

# 7. Command to run the application
CMD ["python", "app.py"]