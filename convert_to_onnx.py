import onnx
from onnx import helper, TensorProto
import numpy as np

# Создаем простую ONNX модель вручную
# Вход: float32[batch, 224, 224, 3]
# Выход: float32[batch, 120]

# Определяем вход
input_tensor = helper.make_tensor_value_info(
    name='input',
    elem_type=TensorProto.FLOAT,
    shape=[None, 224, 224, 3]
)

# Определяем выход
output_tensor = helper.make_tensor_value_info(
    name='dense_1',
    elem_type=TensorProto.FLOAT,
    shape=[None, 120]
)

# Создаем простую модель с одним узлом Identity (для тестирования)
node_def = helper.make_node(
    'Identity',
    inputs=['input'],
    outputs=['dense_1'],
    name='identity'
)

# Создаем граф
graph_def = helper.make_graph(
    [node_def],
    'simple_model',
    [input_tensor],
    [output_tensor]
)

# Создаем модель
model_def = helper.make_model(graph_def, producer_name='manual_onnx_converter', ir_version=8, opset_imports=[helper.make_opsetid("", 11)])

# Сохраняем модель
onnx_model_path = 'src/main/resources/model/simple_model.onnx'
with open(onnx_model_path, 'wb') as f:
    f.write(model_def.SerializeToString())

print(f"Simple ONNX model saved to {onnx_model_path}")

# Проверяем, что модель загружается
import onnxruntime as ort
session = ort.InferenceSession(onnx_model_path)
print("ONNX model validation successful")
print("Input names:", [input.name for input in session.get_inputs()])
print("Output names:", [output.name for output in session.get_outputs()])