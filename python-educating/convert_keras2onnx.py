import keras2onnx
import onnxruntime as ort
import numpy as np
from tensorflow import keras

def convert_h5_to_onnx_keras2onnx(h5_path, onnx_path):
    """
    Конвертирует модель из формата H5 в ONNX с использованием keras2onnx
    """
    print(f"Загружаем модель из {h5_path}...")
    model = keras.models.load_model(h5_path)
    print("Модель загружена успешно")

    # Проверяем архитектуру модели
    print("Архитектура модели:")
    model.summary()

    print(f"Конвертируем в ONNX...")
    # Конвертируем в ONNX с помощью keras2onnx
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # Сохраняем ONNX модель
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

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
        convert_h5_to_onnx_keras2onnx(h5_path, onnx_path)
        print("✅ Конвертация завершена успешно!")
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        import traceback
        traceback.print_exc()