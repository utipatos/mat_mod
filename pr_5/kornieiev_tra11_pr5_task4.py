import os

from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import RMSprop
from matplotlib import pyplot as plt

from definitions import CAT_AND_DOGS_SMALL_DIR

cats_and_dogs_dir = CAT_AND_DOGS_SMALL_DIR


def setup_directory_structure() -> tuple:
    """ Отримання шляхів директорій папок з файлами кішок і собак для: train, test, validation."""

    def get_directory_path(folder_name: str):
        res_dir = os.path.join(cats_and_dogs_dir, folder_name)
        # Вивдення загальної кількості зображень
        cats_sub_dir = os.path.join(res_dir, 'cats')
        print(f'total {folder_name} cat images:', len(os.listdir(cats_sub_dir)))
        dogs_sub_dir = os.path.join(res_dir, 'dogs')
        print(f'total {folder_name} dog images:', len(os.listdir(dogs_sub_dir)))
        return res_dir

    train_dir = get_directory_path('train')
    test_dir = get_directory_path('test')
    validation_dir = get_directory_path('validation')

    return train_dir, test_dir, validation_dir


def build_convolutional_neural_network_model():
    """ Побудова згорткової нейронної мережі """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=1e-4),
                  metrics=['acc'])
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
    """ Побудова графіків точності та похибки """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.title('Training and Validation Accuracy')
    plt.plot(epochs, acc, 'bo', label='Точність навчання')
    plt.plot(epochs, val_acc, 'b', label='Точність перевірки')
    plt.legend()

    plt.figure()

    plt.title('Training and Validation Loss')
    plt.plot(epochs, loss, 'bo', label='Похибка навчання')
    plt.plot(epochs, val_loss, 'b', label='Похибка перевірки')
    plt.legend()

    plt.show()


# Виконання програми
train_dir, _, validation_dir = setup_directory_structure()
# Побудова згорткової нейромережі
model = build_convolutional_neural_network_model()
# Створення генераторів
batch_size = 20
train_generator, validation_generator = create_generators(train_dir, validation_dir, batch_size)
# Тренування моделі
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
)
# Збереження моделі у файл
model.save('cats_and_dogs_small_1.keras')
# Побудова грпфіків≈
plot_training_history(history)
