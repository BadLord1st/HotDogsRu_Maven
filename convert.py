import tensorflow as tf
import tf2onnx

# Load the Keras model
model = tf.keras.models.load_model("./src/main/resources/Keras_Model.h5")

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save ONNX model
with open("./src/main/resources/model/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model converted to ONNX successfully!")