import os
from random import randint
from typing import Literal

from keras import layers, Sequential, Input, backend
from keras.src.optimizers import RMSprop
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array, array_to_img
from matplotlib import pyplot as plt

from definitions import CAT_AND_DOGS_SMALL_DIR

cats_and_dogs_dir = CAT_AND_DOGS_SMALL_DIR
train_dir = os.path.join(cats_and_dogs_dir, 'train')
validation_dir = os.path.join(cats_and_dogs_dir, 'validation')
test_dir = os.path.join(cats_and_dogs_dir, 'test')


def get_train_test_validation_paths(animal_type: Literal['cats', 'dogs']) -> tuple:
    train_sub_dir = os.path.join(train_dir, animal_type)
    print(f'total train {animal_type} images:', len(os.listdir(train_sub_dir)))
    test_sub_dir = os.path.join(test_dir, animal_type)
    print(f'total test {animal_type} images:', len(os.listdir(test_sub_dir)))
    validation_sub_dir = os.path.join(validation_dir, animal_type)
    print(f'total validation {animal_type} images:', len(os.listdir(validation_sub_dir)))

    return train_sub_dir, test_sub_dir, validation_sub_dir


def build_convolutional_neural_network_model():
    """ Побудова згорткової нейронної мережі """
    model = Sequential(
        [
            Input(shape=(150, 150, 3)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=512, activation='relu'),
            layers.Dense(units=1, activation='sigmoid')
        ]
    )
    model.summary()
    return model


def create_generators(train_dir, validation_dir, batch_size):
    """ Створення генераторів для навчання та валідації даних."""
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    return train_generator, validation_generator


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


def augment_and_save_images(datagen, img_dir, target_size=(150, 150), batch_size=1, num_images=4):
    """
    Доповнює та відображає зображення за допомогою наданого ImageDataGenerator.
    """
    # Отримуємо список шляхів до файлів зображень
    fnames = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]

    # Вибір одного зображення для розширення довільним чином
    img_index = randint(0, len(fnames) - 1)
    img_path = fnames[img_index]

    # Читання зображення та зміна його розмірів
    img = load_img(img_path, target_size=target_size)

    # Зміна форми на (1, 150, 150, 3)
    x = img_to_array(img)

    # Reshape the array to include the batch dimension
    x = x.reshape((1,) + x.shape)

    # Генерація пакетів випадково перетворених зображень
    i = 0
    for batch in datagen.flow(x, batch_size=batch_size):
        plt.figure(i)
        img_to_show = array_to_img(batch[0])
        plt.imshow(img_to_show)
        plt.savefig(f"image_{i}.png")
        plt.close()
        i += 1
        if i % num_images == 0:
            break


""" Виконання програми MNIST_Згорткова_нейронна_мережа"""
if __name__ == '__main__':
    train_cats_dir, test_cats_dir, validation_cats_dir = get_train_test_validation_paths('cats')
    train_dogs_dir, test_dogs_dir, validation_dogs_dir = get_train_test_validation_paths('dogs')

    backend.clear_session()
    # Побудова моделі згорткової мережі
    network = build_convolutional_neural_network_model()
    network.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
    # Створення генераторів
    batch_size = 20
    train_generator, validation_generator = create_generators(train_dir, validation_dir, batch_size)
    # Тренування моделі
    history = network.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator
    )
    # Збереження моделі у файл
    network.save('cats_and_dogs_small_1.keras')

    # Побудова грпфіків≈
    plot_training_history(history)

    # розширення навчального набору даних для ослаблення ефекту перенавчання
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    augment_and_save_images(datagen, train_cats_dir)
