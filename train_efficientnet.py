import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

# -----------------------------
# Configuration
# -----------------------------
DATA_ROOT = Path("merged_dataset")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.4
DENSE_UNITS = 512
AUGMENT = True
FINE_TUNE_AT = None  # Set to an integer layer index to unfreeze deeper layers

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "efficientnet_b3_best.keras"


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[WARN] Could not set GPU memory growth: {e}")


def build_datasets():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Train/val directories not found under '{DATA_ROOT}'. "
            "Run merge_datasets.py first."
        )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Detected {len(class_names)} classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    if AUGMENT:
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
            ],
            name="data_augmentation",
        )
    else:
        data_augmentation = keras.Sequential(name="data_augmentation")

    return train_ds, val_ds, len(class_names), data_augmentation


def build_model(num_classes: int, data_augmentation: keras.Sequential) -> keras.Model:
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b3_custom")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())
    return model, base_model


def maybe_fine_tune(base_model: keras.Model) -> None:
    if FINE_TUNE_AT is None:
        return

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    for layer in base_model.layers[FINE_TUNE_AT:]:
        layer.trainable = True
    print(f"[INFO] Enabled fine-tuning from layer index {FINE_TUNE_AT}.")


def train():
    configure_gpu()
    train_ds, val_ds, num_classes, data_aug = build_datasets()
    model, base_model = build_model(num_classes, data_aug)
    maybe_fine_tune(base_model)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    final_model_path = MODEL_DIR / "efficientnet_b3_final.keras"
    model.save(final_model_path)
    print(f"[INFO] Training complete. Final model saved to {final_model_path}")
    print(f"[INFO] Best checkpoint saved to {CHECKPOINT_PATH}")
    return history


if __name__ == "__main__":
    train()



