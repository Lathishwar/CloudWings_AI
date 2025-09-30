# Use the official Python image
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Install system libraries needed by OpenCV and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0
# --------------------

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Copy the rest of your app files
COPY . .

# Expose the port your app runs on
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]