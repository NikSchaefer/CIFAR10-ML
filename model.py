import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
)
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

EPOCHS = 10
learning_rate = 0.001
bs = 32


SAVE_PATH = "save/model"

model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3),
    )
)
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))

opt = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    batch_size=bs,
)

loss, acc = model.evaluate(test_images, test_labels)

print("\nAccuracy: ", acc)

model.save(SAVE_PATH)

"""
(conv2d(32 => 64 => 128) x 2, MaxPooling2d) x 3, Flatten, Dense, Dropout, Dense
epochs: 10
acc: 0.7709000110626221
"""
