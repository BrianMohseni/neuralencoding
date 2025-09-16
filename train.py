import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models


DATASET_DIR = "dataset"
EEG_CHANNELS = 4
FNIRS_CHANNELS = 4
BANDS = ["alpha", "beta", "gamma", "delta", "theta"]
BAND_CHANNELS = len(BANDS) * EEG_CHANNELS
TOTAL_FEATURES = EEG_CHANNELS + FNIRS_CHANNELS + BAND_CHANNELS

EPOCH_SECONDS = 1.0
SAMPLING_RATE = 256
EPOCH_SIZE = int(EPOCH_SECONDS * SAMPLING_RATE)
BATCH_SIZE = 32
EPOCHS = 1


def make_file_list(class_names):
    """Collect (file, label) pairs for splitting at the file level."""
    file_list = []
    label_map = {cls: i for i, cls in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(DATASET_DIR, cls)
        for csv_file in glob.glob(os.path.join(cls_dir, "*.csv")):
            file_list.append((csv_file, label_map[cls]))
    return file_list, label_map


def process_files(files):
    """Turn CSV files into epochs + labels."""
    X_out, y_out = [], []
    for csv_file, label in files:
        df = pd.read_csv(csv_file)

        eeg_cols = [f"eeg_{i}" for i in range(EEG_CHANNELS)]
        fnirs_cols = [f"optics_{i}" for i in range(FNIRS_CHANNELS)]
        band_cols = [f"{band}_{i}" for band in BANDS for i in range(EEG_CHANNELS)]

        features = df[eeg_cols + fnirs_cols + band_cols].values.astype(np.float32)
        n_samples = features.shape[0]

        for start in range(0, n_samples - EPOCH_SIZE, EPOCH_SIZE):
            window = features[start:start + EPOCH_SIZE]
            if window.shape[0] == EPOCH_SIZE:
                X_out.append(window)
                y_out.append(label)

    return np.array(X_out), np.array(y_out)


def load_data():
    class_names = sorted(os.listdir(DATASET_DIR))
    file_list, label_map = make_file_list(class_names)

    # split at the file level, not by epoch
    train_files, val_files = train_test_split(
        file_list,
        test_size=0.2,
        stratify=[lbl for _, lbl in file_list],
        random_state=42,
    )

    X_train, y_train = process_files(train_files)
    X_val, y_val = process_files(val_files)

    return X_train, y_train, X_val, y_val, class_names


def build_model(input_shape, num_classes, X_train=None):
    inputs = layers.Input(shape=input_shape)

    normalizer = layers.Normalization(axis=-1)
    if X_train is not None and len(X_train) > 0:
        normalizer.adapt(X_train)

    x = normalizer(inputs)
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)  # one vector per epoch
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, class_names = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Classes: {class_names}")

    # handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)

    model = build_model(
        input_shape=(EPOCH_SIZE, TOTAL_FEATURES),
        num_classes=len(class_names),
        X_train=X_train
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict
    )

    model.save("model.keras")
    print("Model saved as model.keras")


if __name__ == "__main__":
    main()
