from random import randint
from sys import maxsize
from typing import Literal, Tuple

from keras import layers, Sequential, backend
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from numpy import set_printoptions, ndarray, argmax, expand_dims
import matplotlib.pyplot as plt  # Імпорт бібліотеки графіки


def print_image_attributes(images: ndarray, image_choice: Literal["Training", "Test"]):
    """
     Функція для виведення на екран атрибутів зображень
    :param images:  масив зображень
    :param image_choice: тип зображень 'Training' або 'Test' для виведення даних на екран
    """
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


def plot_training_history(history):
    def build_chart(title, val1, label1, val2, label2):
        epochs = range(1, len(val1) + 1)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.title(title)
        plt.plot(epochs, val1, 'g-', label=label1)
        plt.plot(epochs, val2, 'b--', label=label2)
        plt.legend()
        plt.savefig(f"{label1}.png")
        plt.close()

    """ Побудова графіків точності та похибки """
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']

    build_chart(title='Графік точності розпізнавання зображень',
                val1=acc, label1='Точність навчання (training acc)',
                val2=val_acc, label2='Точність перевірки (test acc)')

    build_chart(title='Графік похибки розпізнавання зображень',
                val1=loss, label1='Похибка навчання (training loss)',
                val2=val_loss, label2='Похибка перевірки (test loss)')


""" Виконання програми """
if __name__ == '__main__':
    # Завантаження даниз з бази MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Характеристики тестових зображень та зображень для навчання
    print_image_attributes(train_images, image_choice="Training")
    print_image_attributes(test_images, image_choice="Test")
    # print(">>> Setting print options to display large arrays without truncation")
    set_printoptions(threshold=maxsize)  # Встановлення режиму можливості повного виводу на екран великих масивів

    # Вибір зображення цифри для навчання під індексом 5 та вивід на екран його числових даних
    digit = train_images[5]
    # print(f">>> Was chosen train image with index 5\n. The digit is:\n")
    # print("\n".join(" ".join(f"{pixel:3d}" for pixel in row) for row in
    #                 digit))  # Такий спосіб дозволяє вивести на екран весь рядок матиці і наглядно побачити цифру

    # Вивід зображення на екран з використанням matplotlib
    # plt.imshow(digit, cmap='binary')
    # print(">>> Showing the selected train image using a binary color map")
    # plt.savefig("Train_image_digit.png")
    # plt.close()

    # Очищення попереднії сесій
    backend.clear_session()

    """ Послідовна структура шарів нейронної мережі """
    network = Sequential()
    # Два щільні шари нейронної мережі зі збільшеною кількістю нейронів
    network.add(layers.Dense(units=1024, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(units=10, activation='softmax'))
    print(f"Summary of neutral network model: \n{network.summary()}")

    # Метод та параметри оптимізації при навчанні
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Зміна формату даних зображень для навчання та тестових зображень
    train_images_reshaped = reshape_images(images=train_images, image_choice="Training", num_images=60000)
    test_images_reshaped = reshape_images(images=test_images, image_choice="Test", num_images=10000)

    # Зміна формату міток (1 у відповідній позиції, решта нулі)
    # Ці мітки (числа від 0 до 9) перетворюються у вектори з 10 компонентами, у яких в позиції, що відповідає числу,
    # знаходиться 1, а в усіх інших позиціях нулі.
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Навчання нейронної мережі
    print(">>> Training of neutral network")
    history = network.fit(train_images_reshaped, train_labels, epochs=5, batch_size=4096,
                          validation_data=(test_images_reshaped, test_labels))

    # Визначення похибок та точності розпізнавання для зображень навчання
    evaluate_network_performance(network, images=train_images_reshaped, labels=train_labels, image_choice="Training")
    evaluate_network_performance(network, images=test_images_reshaped, labels=test_labels, image_choice="Test")

    # Побудова грпфіків
    plot_training_history(history)
