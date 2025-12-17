# Инструкции по переносу обучения на сервер

## 1. Подготовка файлов для сервера

### Скопируйте на сервер:
```bash
# Весь проект
scp -r /Users/badwolf/projects/HotDogsRu_Maven user@server:/path/to/projects/

# Или только папку с обучением
scp -r /Users/badwolf/projects/HotDogsRu_Maven/python-educating user@server:/path/to/
```

## 2. Настройка сервера

### Установите uv (если нет):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Установите CUDA (для GPU):
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Проверьте установку
nvidia-smi
```

## 3. Запуск обучения

### Перейдите в папку проекта:
```bash
cd /path/to/HotDogsRu_Maven/python-educating
```

### Запустите обучение:
```bash
# Автоматический скрипт
chmod +x train_server.sh
./train_server.sh

# Или вручную
uv run python train.py --data-root archive/images/Images --epochs 50 --batch_size 32 --export-onnx
```

## 4. Мониторинг обучения

### В отдельном терминале мониторьте:
```bash
# GPU использование
watch nvidia-smi

# Процессы Python
ps aux | grep python

# Использование диска
df -h
```

## 5. После обучения

### Скачайте готовую модель обратно:
```bash
# С локальной машины
scp user@server:/path/to/HotDogsRu_Maven/src/main/resources/model/* /Users/badwolf/projects/HotDogsRu_Maven/src/main/resources/model/
```

## Параметры обучения

- `--epochs 50`: 50 эпох обучения
- `--batch_size 32`: размер батча
- `--export-onnx`: автоматический экспорт в ONNX
- `--data-root`: путь к датасету

## Ожидаемое время обучения

- На GPU RTX 3090: ~2-4 часа
- На CPU: ~10-20 часов
- На H100: ~30-60 минут

## Troubleshooting

### Если CUDA не работает:
```bash
# Установите tensorflow для CUDA
uv add tensorflow[and-cuda]
```

### Если память кончается:
```bash
# Уменьшите batch_size
uv run python train.py --batch_size 16 --epochs 50
```

### Если обучение прерывается:
```bash
# Модель сохранится автоматически в interrupted_model.keras
# Продолжите с чекпоинта
uv run python train.py --resume interrupted_model.keras
```