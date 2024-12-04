# Use the official Python image from Docker Hub
FROM python:3.10.6-buster

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY API/api_test.py /app/api_test.py
COPY API/requirements.txt /app/requirements.txt
COPY FakeNews_packages/preprocessing2.py /app/FakeNews_packages/
COPY FakeNews_packages/model_ml.py /app/FakeNews_packages/
COPY FakeNews_packages/model_dl.py /app/FakeNews_packages/
COPY FakeNews_packages/params.py /app/FakeNews_packages/

# Install dependencies (including FastAPI and Uvicorn)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn when the container starts
CMD ["uvicorn", "api_test:app", "--host", "0.0.0.0", "--port", "8080"]
