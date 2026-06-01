import tensorflow as tf
from tensorflow.keras import layers, models

# PNEUMONIA DETECTOR from chest X-rays.
# a convolutional neural network that looks at a chest x-ray and says whether it
# shows pneumonia or a normal lung. trained from scratch on grayscale x-rays.
#
# DATA: the classic "Chest X-Ray Images (Pneumonia)" dataset from Kaggle works.
# layout: data/train/NORMAL, data/train/PNEUMONIA, data/val/...
# pip install tensorflow

IMG = 150
BATCH = 32


def make_model():
    # 3 conv blocks then a dense classifier. x-rays are grayscale (1 channel).
    return models.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # prob of pneumonia
    ])


def main():
    train = tf.keras.utils.image_dataset_from_directory(
        "data/train", image_size=(IMG, IMG), batch_size=BATCH,
        color_mode="grayscale", label_mode="binary")
    val = tf.keras.utils.image_dataset_from_directory(
        "data/val", image_size=(IMG, IMG), batch_size=BATCH,
        color_mode="grayscale", label_mode="binary")

    model = make_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(train, validation_data=val, epochs=8)

    model.save("pneumonia_model.keras")
    print("saved -> pneumonia_model.keras")


if __name__ == "__main__":
    main()
