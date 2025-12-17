import logging
import os
from typing import List
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to where the app is run, or absolute. 
# Since we will run from python_app root, and code is in app/model.py,
# we need to be careful. Best to use absolute paths relative to this file.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.onnx")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

classes: List[str] = []
ort_session = None

CLASS_MAPPING = {
    "beagle": "Beagle",
    "Pembroke": "Corgi",
    "Cardigan": "Corgi",
    "Blenheim_spaniel": "Blenheim_spaniel",
    "dalmatian": "Dalmatian",
    "German_shepherd": "German_shepherd",
    "Siberian_husky": "Huskies",
    "Labrador_retriever": "Labrador",
    "shar_pei": "Shar_pei"
}

def load_model():
    global ort_session, classes
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        with open(CLASSES_PATH, "r") as f:
            for line in f:
                parts = line.strip().split("-")
                if len(parts) >= 2:
                    breed_name = parts[1]
                    classes.append(breed_name)
        logger.info(f"Loaded {len(classes)} classes.")
    except Exception as e:
        logger.error(f"Error loading model or classes: {e}")

def preprocess_image(image_data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Normalize to [0, 1]
        img_array /= 255.0
        
        # Add batch dimension: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def get_prediction(image_data: bytes) -> str:
    if ort_session is None:
        load_model()
    
    if ort_session is None:
         raise RuntimeError("Model could not be loaded")

    input_data = preprocess_image(image_data)
    
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_data})
    
    output = outputs[0]
    if len(output.shape) > 1:
        output = output[0]
        
    max_index = np.argmax(output)
    
    if 0 <= max_index < len(classes):
        raw_class = classes[max_index]
        return CLASS_MAPPING.get(raw_class, raw_class)
    return "Unknown"
