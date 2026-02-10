# IA App for CNN classification with fastapi and streamlit

## Description
This project is a web application that allows users to upload images and classify them using a pre-trained CNN model. The application is built using FastAPI for the backend and Streamlit for the frontend.


## Project Structure
```.
├── app
│   ├── main.py
│   ├── model.py
│   ├── preprocess.py
├── examples
│   ├── examples/mnist_cnn_model.ipynb
├── model
│   ├── mnist_cnn_model.keras
├── streamlit
│   ├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```

## Installation
1. Clone the repository:
    ```bash
    git clone git@github.com:jkarsenty/fastapi-streamlit-ia-app.git
    cd fastapi-streamlit-ia-app
    ```

2. Install the docker image :
    ```bash
    docker build -t fastapi-streamlit-ia-app .
    ```

3. Run the docker container:
    ```bash
    docker run -p 8000:8000 fastapi-streamlit-ia-app
    ```

4. Open your browser and go to `http://localhost:8000` to access the FastAPI interface or `http://localhost:8501` for the Streamlit app.


## Usage
1. **FastAPI Interface**:
   - Navigate to `http://localhost:8000/docs` to access the Swagger UI.
   - You can upload an image and get the classification result. 

2. **Streamlit App**:
   - Navigate to `http://localhost:8501` to access the Streamlit app.
   - You can upload an image and get the classification result displayed on the page.

## Example
You can find an example of how to create and train a CNN model in the `examples/mnist_cnn_model.ipynb` notebook. This notebook demonstrates how to preprocess the MNIST dataset, build a CNN model, train it, and save the model for later use.
