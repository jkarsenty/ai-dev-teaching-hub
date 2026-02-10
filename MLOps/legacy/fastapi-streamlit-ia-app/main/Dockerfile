# Dockerfile for building a Docker image with Python 3.11 for a fastapi app and streamlit app
# It installs the necessary dependencies and sets up the environment for running the apps.

FROM python:3.11-slim

# Set environment variables to avoid Python buffering and to ensure the output is sent directly to the terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8  

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Expose the port that the Streamlit app will run on
EXPOSE 8501

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0", "--port", "8000", "--reload"]

# Command to run the Streamlit app
# Uncomment the line below if you want to run the Streamlit app instead of the FastAPI
# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0"]

# Note: The CMD line for Streamlit is commented out. You can uncomment it to run the Streamlit app instead of the FastAPI app.

# To build the Docker image, run the following command in the terminal:
# docker build -t my_fastapi_streamlit_app .
# To run the Docker container, use the following command:
# docker run -p 8000:8000 -p 8501:8501 my_fastapi_streamlit_app
# This will map the container's ports to the host machine's ports, allowing you to access the FastAPI app at http://localhost:8000 and the Streamlit app at http://localhost:8501.