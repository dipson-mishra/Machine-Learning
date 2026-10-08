# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy dependency files first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the app using Waitress (ideal for cross-platform/Windows compatibility)
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "src.app:app"]