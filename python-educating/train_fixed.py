#!/usr/bin/env python3
"""
Исправленный скрипт обучения для Stanford Dogs с правильными настройками:
- Разрешение 224x224 (стандарт для MobileNetV2)
- Аугментация данных для предотвращения переобучения
- Transfer learning с MobileNetV2
"""

import argparse
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def find_class_dirs(data_root, keywords=None):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    dirs = [p.name for p in sorted(data_root.iterdir()) if p.is_dir()]
    if not keywords:
        return dirs
    keywords = [k.strip().lower() for k in keywords]
    matched = []
    for d in dirs:
        low = d.lower()
        for k in keywords:
            if k in low:
                matched.append(d)
                break
    return matched


def build_model_with_augmentation(num_classes, img_height=224, img_width=224):
    """Строим модель с аугментацией данных и правильной архитектурой"""
    model = Sequential([
        # Аугментация данных
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        # Предобработка
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        # Transfer learning с MobileNetV2
        keras.applications.MobileNetV2(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        ),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes)
    ])

    # Замораживаем базовую модель
    model.layers[4].trainable = False  # MobileNetV2 слой

    return model


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(__doc__))
    parser.add_argument('--data-root', required=True, help='Root folder with class subfolders')
    parser.add_argument('--selected', help='Comma-separated keywords to match class folder names')
    parser.add_argument('--list-classes', action='store_true', help='Only list discovered classes and exit')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=224, help='Image size (224 recommended for MobileNetV2)')
    parser.add_argument('--output', default='src/main/resources/model/interrupted_model.keras')
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print('Data root does not exist:', data_root)
        sys.exit(1)

    keywords = None
    if args.selected:
        keywords = [k.strip() for k in args.selected.split(',') if k.strip()]

    classes = find_class_dirs(data_root, keywords)
    if not classes:
        print('No classes found for keywords:', keywords)
        sys.exit(1)

    print(f'Found classes ({len(classes)}):')
    for c in classes:
        print(f'  {c}')

    if args.list_classes:
        print('\nList-only mode, exiting.')
        return

    # Создаем временную директорию только с выбранными классами
    temp_dir = tempfile.mkdtemp()
    print(f'Using temp dir: {temp_dir}')
    try:
        for cls in classes:
            src = data_root / cls
            dst = Path(temp_dir) / cls
            if src.exists():
                shutil.copytree(src, dst)
        data_root = Path(temp_dir)

        # Настраиваем производительность для Apple Silicon
        try:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            print('Mixed precision enabled: mixed_float16')
        except Exception:
            pass

        img_height = args.img_size
        img_width = args.img_size

        print('Preparing training and validation datasets...')
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_root,
            labels='inferred',
            label_mode='int',
            color_mode='rgb',
            batch_size=args.batch_size,
            image_size=(img_height, img_width),
            shuffle=True,
            seed=123,
            validation_split=args.val_split,
            subset='training'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_root,
            labels='inferred',
            label_mode='int',
            color_mode='rgb',
            batch_size=args.batch_size,
            image_size=(img_height, img_width),
            shuffle=True,
            seed=123,
            validation_split=args.val_split,
            subset='validation'
        )

        # Оптимизация производительности
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Строим модель с аугментацией
        model = build_model_with_augmentation(num_classes=len(classes), img_height=img_height, img_width=img_width)

        # Компилируем
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )

        model.summary()

        print(f'Starting training: epochs={args.epochs}, batch_size={args.batch_size}, img_size={args.img_size}x{args.img_size}')

        # Callbacks для лучшего обучения
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks
        )

        # Сохраняем модель
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        print(f'Saved Keras model to {out_path}')

        # Выводим финальные метрики
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(".2f")
        print(".2f")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()