# Use the official Python image from Docker Hub
FROM python:3.10.6-buster



# Copy the application code to the container
COPY api_test.py /api_test.py
COPY requirements.txt /requirements.txt

# Install dependencies (including FastAPI and Uvicorn)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn when the container starts
CMD ["uvicorn", "api_test:app", "--host", "0.0.0.0", "--port", "8080"]
