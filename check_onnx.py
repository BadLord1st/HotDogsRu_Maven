import onnxruntime as ort

# Load the ONNX model
session = ort.InferenceSession('./src/main/resources/model/model.onnx')

# Get input names
input_names = [input.name for input in session.get_inputs()]
print("Input names:", input_names)

# Get output names
output_names = [output.name for output in session.get_outputs()]
print("Output names:", output_names)