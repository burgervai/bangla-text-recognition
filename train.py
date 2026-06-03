import os
import json
import argparse
import numpy as np
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import cv2

IMG_SIZE = 64
BATCH_SIZE = 32

DATA_DIR = r"C:\Users\DataInsight\Downloads\bangla-ocr-assignment\BanglaLekha-Isolated\Images"

MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"


def load_dataset(data_dir):
    images = []
    labels = []

    data_path = Path(data_dir)

    class_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir()],
        key=lambda x: int(x.name)
    )

    for class_dir in class_dirs:
        print(f"Loading class {class_dir.name}")

        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.bitwise_not(img)

            images.append(img)
            labels.append(class_dir.name)

    return np.array(images), np.array(labels)


def preprocess(images):
    x = images.astype("float32") / 255.0
    return x[..., np.newaxis]


def build_model(num_classes, use_transfer=False):

    if use_transfer:

        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )

        base.trainable = False

        inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))

        x = layers.Conv2D(3, 1, padding="same")(inp)

        x = base(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation="relu")(x)

        x = layers.Dropout(0.3)(x)

        out = layers.Dense(
            num_classes,
            activation="softmax"
        )(x)

        return models.Model(inp, out)

    else:

        return models.Sequential([

            layers.Input((IMG_SIZE, IMG_SIZE, 1)),

            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),

            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),

            layers.Dense(
                num_classes,
                activation="softmax"
            )

        ])


def train(use_transfer=False, epochs=2, lr=1e-3):

    print("Loading dataset...")
    images, labels = load_dataset(DATA_DIR)

    print(f"Loaded {len(images)} samples")
    print(f"Classes: {len(set(labels))}")

    le = LabelEncoder()

    y = le.fit_transform(labels)

    label_map = {
        int(i): str(c)
        for i, c in enumerate(le.classes_)
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            label_map,
            f,
            ensure_ascii=False,
            indent=4
        )

    x = preprocess(images)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    num_classes = len(le.classes_)

    model = build_model(
        num_classes,
        use_transfer
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    run_name = (
        f"{'transfer' if use_transfer else 'cnn'}"
        f"_lr{lr}_ep{epochs}"
    )

    mlflow.set_experiment("bangla-ocr")

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model_type": "MobileNetV2" if use_transfer else "CNN",
            "epochs": epochs,
            "lr": lr,
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "num_classes": num_classes
        })

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )

        datagen.fit(x_train)

        os.makedirs("models", exist_ok=True)

        cb = [

            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            ),

            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=3,
                factor=0.5,
                verbose=1
            ),

            callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            )
        ]

        history = model.fit(
            datagen.flow(
                x_train,
                y_train,
                batch_size=BATCH_SIZE
            ),
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=cb,
            verbose=1
        )

        train_acc = max(history.history["accuracy"])
        train_loss = min(history.history["loss"])

        val_acc = max(history.history["val_accuracy"])
        val_loss = min(history.history["val_loss"])

        mlflow.log_metrics({
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss)
        })

        plt.figure(figsize=(8, 4))

        plt.plot(
            history.history["accuracy"],
            label="train"
        )

        plt.plot(
            history.history["val_accuracy"],
            label="validation"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        plt.savefig("accuracy_curve.png")
        plt.close()

        mlflow.log_artifact("accuracy_curve.png")

        preds = model.predict(
            x_val,
            verbose=0
        )

        preds = np.argmax(
            preds,
            axis=1
        )

        report = classification_report(
            y_val,
            preds,
            output_dict=True
        )

        with open(
            "classification_report.json",
            "w"
        ) as f:
            json.dump(
                report,
                f,
                indent=4
            )

        mlflow.log_artifact(
            "classification_report.json"
        )

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(LABELS_PATH)

        print()
        print("=" * 50)
        print(f"Best Validation Accuracy: {val_acc:.4f}")
        print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transfer",
        action="store_true"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    train(
        args.transfer,
        args.epochs,
        args.lr
    )