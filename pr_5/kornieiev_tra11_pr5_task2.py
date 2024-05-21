import sys
from typing import Literal, Tuple

import matplotlib.pyplot as plt
from keras import layers, Sequential
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from numpy import set_printoptions, ndarray

def print_image_attributes(images: ndarray, image_choice: Literal["Training", "Test"]):
    print(f"Attributes of {image_choice} Images:")
    print(f"Number of dimension: {images.ndim}")  # Розмірність зображень
    print(f"Data type: {images.dtype}")  # Тип даних зображень
    print(f"Shape of the tensor: {images.shape}")  # Формат зображень
    print(f"Number of training labels: {len(images)}")  # Розмір масиву міток


def reshape_images(images: ndarray,
                   image_choice: Literal["Training", "Test"],
                   num_images: int,
                   img_shape: Tuple[int, int] = (28, 28),
                   normalization_factor: float = 255.0
                   ) -> ndarray:
    """
     Зміна формату даних зображень

     :param images: масив зображень
     :param image_choice: тип зображень для логування в консоль ('Training' або 'Test')
     :param num_images: Кількість зображень для зміни формату.
     :param img_shape: Вихідна форма кожного зображення (висота, ширина).
      normalization_factor: Коефіцієнт нормалізації значень пікселів (за замовчуванням 255 для зображень у градаціях сірого)
     """
    print(f">>> Reshape and normalize the {image_choice} images data")
    reshaped_images = images.reshape((num_images, img_shape[0] * img_shape[1]))
    return reshaped_images.astype('float32') / normalization_factor


def evaluate_network_performance(network: Sequential,
                                 images: ndarray, labels: ndarray,
                                 image_choice: Literal["Training", "Test"]):
    """
    Визначення похибок та точності розпізнавання

    :param network: нейронна мережа
    :param images: масив зображень
    :param labels: мітки зображень
    :param image_choice: тип зображень для логування в консоль ('Training' або 'Test')
    """
    print(f">>> Evaluating the network's performance on the {image_choice} images")
    test_loss, test_acc = network.evaluate(images, labels)
    print(f"{image_choice} images: Loss: {test_loss}\n"
          f"{image_choice} image: Accuracy: {test_acc}")

""" Виконання програми """
# Завантаження даниз з бази MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Характеристики тестових зображень та зображень для навчання
print_image_attributes(train_images, image_choice="Training")
print_image_attributes(test_images, image_choice="Test")
print(">>> Setting print options to display large arrays without truncation")
set_printoptions(threshold=sys.maxsize)  # Встановлення режиму можливості повного виводу на екран великих масивів

# Вибір зображення цифри з індексом 4 та вивід на екран його числових даних
selected_number_index: int = 4
digit = train_images[selected_number_index]
print(f">>> Selecting training image with index: {selected_number_index}:\n{digit}")

# Вивід зображення на екран з використанням matplotlib
plt.imshow(digit, cmap='binary')
print(">>> Showing the selected image using a binary color map")
plt.show()

""" Послідовна структура шарів нейронної мережі """
network = Sequential()
# Два щільні шари нейронної мережі з відповідною кількістю нейронів та функціями активації
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
print(f"Summary of neutral network model: \n{network.summary()}")

# Метод та параметри оптимізації при навчанні
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Зміна формату даних зображень для навчання
train_images = reshape_images(images=train_images, image_choice="Training", num_images=60000)
test_images = reshape_images(images=test_images, image_choice="Test", num_images=10000)

# Зміна формату міток (1 у відповідній позиції, решта нулі)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Навчання нейронної мережі
print(">>> Training the neural network")
network.fit(train_images, train_labels, epochs=5, batch_size=4096)

# Визначення похибок та точності розпізнавання
evaluate_network_performance(network, images=train_images, labels=train_labels, image_choice="Training")
evaluate_network_performance(network, images=test_images, labels=test_labels, image_choice="Test")
