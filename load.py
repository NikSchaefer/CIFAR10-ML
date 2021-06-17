from CIFAR10.model import SAVE_PATH
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

SAVE_PATH = ""

model = tf.keras.models.load_model(SAVE_PATH)

# model.predict()