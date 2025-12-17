#!/usr/bin/env python3
"""
Training helper for Stanford Dogs subset.
Usage examples:
  # list available classes
  python train.py --data-root python-educating/archive/images/Images --list-classes

  # train on selected breeds (keyword match)
  python train.py --data-root python-educating/archive/images/Images --selected beagle,pembroke --epochs 10 --batch_size 32

  # train on all classes (warning: very slow)
  python train.py --data-root python-educating/archive/images/Images --epochs 20

Outputs:
  - saves Keras model to src/main/resources/Keras_Model.h5 (by default)
  - optionally converts to ONNX if --export-onnx

This script is tuned for mac M1/M2/M4 using tensorflow-macos/tensorflow-metal when available.
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


def build_model(num_classes, img_height=512, img_width=512, use_transfer=False):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = inputs
    x = layers.Rescaling(1.0 / 255)(x)
    # small conv stack if not transfer
    if not use_transfer:
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(num_classes)(x)
        model = keras.Model(inputs, outputs)
        return model
    else:
        # MobileNetV2 transfer learning (faster/accurate)
        base = keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes)(x)
        model = keras.Model(inputs, outputs)
        return model


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(__doc__))
    parser.add_argument('--data-root', required=True, help='Root folder with class subfolders (ImageNet-style)')
    parser.add_argument('--selected', help='Comma-separated keywords to match class folder names (e.g. beagle,pembroke)')
    parser.add_argument('--list-classes', action='store_true', help='Only list discovered classes and exit')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--output', default='src/main/resources/Keras_Model.h5')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--use_transfer', action='store_true', help='Use MobileNetV2 transfer learning')
    parser.add_argument('--export-onnx', action='store_true', help='Export ONNX after training (requires onnxmltools/tf2onnx)')
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

    print('Found classes (%d):' % len(classes))
    for c in classes:
        print('  ', c)

    if args.list_classes:
        print('\nList-only mode, exiting.')
        return

    # Create temporary directory with only selected classes
    temp_dir = tempfile.mkdtemp()
    print('Using temp dir:', temp_dir)
    try:
        for cls in classes:
            src = data_root / cls
            dst = Path(temp_dir) / cls
            if src.exists():
                shutil.copytree(src, dst)
        data_root = Path(temp_dir)

        # configure performance for Apple silicon if available
        try:
            # Enable mixed precision if supported
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

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model = build_model(num_classes=len(classes), img_height=img_height, img_width=img_width, use_transfer=args.use_transfer)

        # compile
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        model.summary()

        print('Starting training: epochs=%d batch_size=%d' % (args.epochs, args.batch_size))
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        # save
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        print('Saved Keras model to', out_path)

        if args.export_onnx:
            try:
                import tf2onnx
                import onnx
                # export using concrete function
                full_model = tf.function(lambda x: model(x))
                full_model = full_model.get_concrete_function(tf.TensorSpec((None, img_height, img_width, 3), tf.float32))
                onnx_model, _ = tf2onnx.convert.from_function(full_model, input_signature=[tf.TensorSpec((None, img_height, img_width, 3), tf.float32)], opset=13)
                with open('src/main/resources/model/model.onnx', 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                print('Exported ONNX to src/main/resources/model/model.onnx')
            except Exception as e:
                print('ONNX export failed:', e)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
