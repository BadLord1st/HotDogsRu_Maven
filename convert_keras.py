import onnxmltools
import tensorflow as tf

# Try to load Keras model
try:
    model = tf.keras.models.load_model('./src/main/resources/Keras_Model.h5')
    print("Loaded Keras model")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit(1)

# Convert to ONNX
onnx_model = onnxmltools.convert_keras(model)

# Save ONNX model
onnxmltools.utils.save_model(onnx_model, './src/main/resources/model/model.onnx')
print("Converted to ONNX")