import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils_emnist import load_images, load_labels, fix_orientation

BASE_DIR = Path(__file__).resolve().parent
EMNIST_DIR = BASE_DIR / "emnist" / "byclass"
MODEL_PATH = BASE_DIR / "model" / "emnist_cnn_new.h5"

try:
    if not EMNIST_DIR.exists():
        raise FileNotFoundError(f"EMNIST data directory not found at {EMNIST_DIR}")
    
    print("Loading training data...")
    X_train = load_images(str(EMNIST_DIR / "emnist-byclass-train-images-idx3-ubyte"))
    y_train = load_labels(str(EMNIST_DIR / "emnist-byclass-train-labels-idx1-ubyte"))
    
    print("Loading test data...")
    X_test = load_images(str(EMNIST_DIR / "emnist-byclass-test-images-idx3-ubyte"))
    y_test = load_labels(str(EMNIST_DIR / "emnist-byclass-test-labels-idx1-ubyte"))
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

X_train = fix_orientation(X_train) / 255.0
X_test = fix_orientation(X_test) / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    GlobalAveragePooling2D(),

    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(62, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

try:
    print("Starting model training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        validation_data=(X_test, y_test),
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved as HDF5: {MODEL_PATH}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH).replace('.h5', '_interrupted.h5'))
    print("Interrupted model saved")
except Exception as e:
    print(f"Error during training: {e}")
