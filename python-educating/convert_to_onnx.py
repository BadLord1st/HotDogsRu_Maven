import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
from tensorflow import keras

def convert_h5_to_onnx(h5_path, onnx_path, input_shape=(512, 512, 3)):
    """
    Конвертирует модель из формата H5 в ONNX
    """
    print(f"Загружаем модель из {h5_path}...")
    model = keras.models.load_model(h5_path)
    print("Модель загружена успешно")

    # Проверяем архитектуру модели
    print("Архитектура модели:")
    model.summary()

    # Создаем спецификацию входа для ONNX
    spec = (tf.TensorSpec((None, input_shape[0], input_shape[1], input_shape[2]), tf.float32, name="input"),)

    print(f"Конвертируем в ONNX с входом {input_shape}...")
    # Конвертируем в ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Сохраняем ONNX модель
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Модель сохранена в {onnx_path}")

    # Проверяем ONNX модель
    print("Проверяем ONNX модель...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    print(f"Вход ONNX модели: {input_name}")
    print(f"Форма входа: {session.get_inputs()[0].shape}")

    return True

if __name__ == "__main__":
    h5_path = "/Users/badwolf/projects/HotDogsRu_Maven/from-server/best_model.h5"
    onnx_path = "/Users/badwolf/projects/HotDogsRu_Maven/src/main/resources/model/best_model.onnx"

    try:
        convert_h5_to_onnx(h5_path, onnx_path)
        print("✅ Конвертация завершена успешно!")
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        import traceback
        traceback.print_exc()