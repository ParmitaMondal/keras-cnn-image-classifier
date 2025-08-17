import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# Paths & data
# --------------------------
train_img_path = "new_train/"
test_img_path  = "new_test/"
df_train = pd.read_csv("train.csv")   # expects columns: image_name, target
df_test  = pd.read_csv("test.csv")    # expects column: image_name

IMG_SIZE = (128, 128)

def load_and_resize(img_path, size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

# --- Load train images ---
train_files = [os.path.join(train_img_path, f"{name}.jpg") for name in df_train["image_name"]]
X_train = np.stack([load_and_resize(p) for p in train_files]).astype("float32") / 255.0

# Labels -> one-hot
y_int = df_train["target"].astype(int).values
num_classes = int(max(2, y_int.max() + 1))
y_train_oh = keras.utils.to_categorical(y_int, num_classes=num_classes)

print(f"X_train: {X_train.shape}, y: {y_train_oh.shape} (classes={num_classes})")

# --------------------------
# Model
# --------------------------
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------
# Callbacks (save best weights)
# --------------------------
model_name = "CNN"
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)
pweight = os.path.join(weights_dir, f"weights_{model_name}.h5")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=pweight, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )
]

# --------------------------
# Train
# --------------------------
history = model.fit(
    X_train, y_train_oh,
    epochs=15,
    batch_size=100,
    validation_split=0.1,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)
print(f"CNN weights saved in {pweight}")

# --------------------------
# Load best weights (optional if EarlyStopping restored already)
# --------------------------
model.load_weights(pweight)

# --------------------------
# Test set predictions
# --------------------------
test_files = [os.path.join(test_img_path, f"{name}.jpg") for name in df_test["image_name"]]
X_test = np.stack([load_and_resize(p) for p in test_files]).astype("float32") / 255.0

pred_probs = model.predict(X_test, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
df_test["predicted_target"] = pred_labels

# Save CSV
out_csv = "CNN_outcome.csv"
df_test.to_csv(out_csv, index=False)
print(f"Saved predictions to {out_csv}")

# --------------------------
# Plots
# --------------------------
epochs_run = range(len(history.history["loss"]))

plt.figure()
plt.plot(epochs_run, history.history["loss"], "r", label="Training loss")
plt.plot(epochs_run, history.history["val_loss"], "b", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

plt.figure()
plt.plot(epochs_run, history.history["accuracy"], "r", label="Training accuracy")
plt.plot(epochs_run, history.history["val_accuracy"], "b", label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()
