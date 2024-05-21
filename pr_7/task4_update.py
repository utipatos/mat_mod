from random import randint
from typing import Literal

from keras import layers, Sequential, Input, backend
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import ndarray, expand_dims, argmax


def build_convolutional_neural_network_model():
    """ Побудова згорткової нейронної мережі """
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(units=64, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ]
    )
    model.summary()
    return model


def reshape_images(images: ndarray,
                   image_choice: Literal["Training", "Test"],
                   num_images: int,
                   img_shape: tuple,
                   normalization_factor: float = 255.0
                   ) -> ndarray:
    """
     Зміна формату даних зображень та зміна формату міток (1 у відповідній позиції, решта нулі)

     :param images: масив зображень
     :param image_choice: тип зображень для логування в консоль ('Training' або 'Test')
     :param num_images: Кількість зображень для зміни формату.
     :param img_shape: Вихідна форма кожного зображення (висота, ширина).
      normalization_factor: Коефіцієнт нормалізації значень пікселів (за замовчуванням 255 для зображень у градаціях сірого)
     """
    print(f">>> Reshape and normalize the {image_choice} images data")
    reshaped_images = images.reshape((num_images, img_shape[0], img_shape[1], 1))
    return reshaped_images.astype('float32') / normalization_factor


def evaluate_network_performance(network: Sequential, images: ndarray, labels: ndarray, image_choice: str):
    """
    Оцінка продуктивності нейронної мережі на даних зображеннях

    :param network: нейронна мережа для оцінки
    :param images: дані зображень для оцінки
    :param labels: правильні мітки для оцінки
    :param image_choice: тип зображень для логування в консоль ('Training' або 'Test')
    """
    print(f">>> Evaluating the network's performance on the {image_choice} images")
    test_loss, test_acc = network.evaluate(images, labels)
    print(f'Loss on {image_choice} images: {test_loss:.4f}')
    print(f'Accuracy on {image_choice} images: {test_acc:.4f}')


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


""" Виконання програми MNIST_Згорткова_нейронна_мережа"""
if __name__ == '__main__':
    # Завантаження даних MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Зміна формату даних зображень для навчання та тестових зображень
    train_images_reshaped = reshape_images(images=train_images, image_choice="Training", num_images=60000,
                                           img_shape=(28, 28, 1))
    test_images_reshaped = reshape_images(images=test_images, image_choice="Test", num_images=10000,
                                          img_shape=(28, 28, 1))

    # Зміна формату міток (1 у відповідній позиції, решта нулі)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    backend.clear_session()
    # Побудова моделі згорткової мережі
    network = build_convolutional_neural_network_model()

    # Метод та параметри оптимізації при навчанні
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Навчання нейронної мережі
    print(">>> Training of neutral network")
    history = network.fit(train_images_reshaped, train_labels, epochs=5, batch_size=64,
                          validation_data=(test_images_reshaped, test_labels))

    # Визначення похибок та точності розпізнавання для зображень навчання
    evaluate_network_performance(network, images=train_images_reshaped, labels=train_labels, image_choice="Training")
    evaluate_network_performance(network, images=test_images_reshaped, labels=test_labels, image_choice="Test")

    # Побудова графіків
    plot_training_history(history)
