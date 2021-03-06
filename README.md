# CIFAR10 Machine Learning Model

Machine Learning Image Classification Model built to classify images of the CIFAR10 dataset

## Layers

The model consists of 13 layers. The first 2 layers consist of 2 Convolutional 2D with 32 filters and relu activation. These layers are intended to draw out the first hint of features. The next layer features a MaxPooling2D layer to bring the image back into size with the features. The following 6 layers repeat a similar process but with more filters increasingly onward. This is due to extracting larger featues over time after smaller features are shown.

The last 4 layers include a Flatten layer to make the data 1 dimensional followed by a Dense layered Decision tree with a dropout layer of 30% near the end. The final layer uses a softmax activation to bring the data back to normal.

```py
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
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
```

## Accuracy

The Model achieved 77% accuracy when evaluating the model on the test set.

## Data

Data is from the CIFAR-10 Dataset located [here](https://www.cs.toronto.edu/~kriz/cifar.html) or loaded with tensorflow datasets

```py
from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```

## Installation

Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
