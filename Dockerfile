FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app.py templates/ .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
