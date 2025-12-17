#!/bin/bash
# Скрипт для автоматического обучения модели на сервере

echo "=== Начинаем обучение модели Stanford Dogs ==="

# Проверяем наличие GPU
echo "Проверка GPU..."
nvidia-smi || echo "GPU не найдена, используем CPU"

# Устанавливаем зависимости через uv
echo "Установка зависимостей..."
uv sync

# Запускаем обучение
echo "Запуск обучения..."
uv run python train.py \
    --data-root archive/images/Images \
    --epochs 50 \
    --batch_size 32 \
    --export-onnx

echo "Обучение завершено!"

# Проверяем результаты
echo "Проверка сохраненных файлов..."
ls -la ../src/main/resources/model/

echo "Готово! Модель обучена и экспортирована в ONNX."