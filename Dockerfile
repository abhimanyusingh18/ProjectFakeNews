# Use Python 3 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('stopwords')"

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
