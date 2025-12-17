import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('./src/main/resources/Keras_Model.h5')

# Print summary
model.summary()

# Print input shape
print("Input shape:", model.input_shape)