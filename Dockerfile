# Dockerfile
# Use a good base image with CUDA support
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Copy files
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for HF Spaces applications
EXPOSE 7860

# Command to run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]