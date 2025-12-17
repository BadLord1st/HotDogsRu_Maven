# HotDogsRu Python App

This is a Python rewrite of the HotDogsRu Java application.

## Prerequisites

- Python 3.8+
- pip

## Installation

1.  Navigate to the `python_app` directory:
    ```bash
    cd python_app
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

1.  Start the server:
    ```bash
    python main.py
    ```
    Or using uvicorn directly:
    ```bash
    uvicorn main:app --reload
    ```

2.  Open your browser and go to `http://localhost:8000`.

## Structure

- `main.py`: The FastAPI application and model inference logic.
- `model/`: Contains the ONNX model and classes file.
- `static/`: Static assets (CSS, JS, Images).
- `templates/`: HTML templates.
