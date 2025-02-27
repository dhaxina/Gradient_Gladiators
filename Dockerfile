# Base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the app files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# Expose the port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
