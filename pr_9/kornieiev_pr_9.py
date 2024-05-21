import os
import ssl
from typing import Literal

import certifi
import numpy as np
from keras import layers, Sequential
from keras.src.applications.vgg16 import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import RMSprop
from matplotlib import pyplot as plt

from definitions import CAT_AND_DOGS_SMALL_DIR

# Створення сетрифікатів для викачення даних моделі VGG16
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Отримання шляхів основних директорій зображень котів та собак
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


def extract_features(directory, sample_count, batch_size):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


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
    # Виділення згорткової основи
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    conv_base.summary()

    train_cats_dir, test_cats_dir, validation_cats_dir = get_train_test_validation_paths('cats')
    train_dogs_dir, test_dogs_dir, validation_dogs_dir = get_train_test_validation_paths('dogs')

    # Виділення ознак за допомогою попередньо навченої згорткової основи
    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20
    train_features, train_labels = extract_features(train_dir, sample_count=2000, batch_size=batch_size)
    validation_features, validation_labels = extract_features(validation_dir, sample_count=1000, batch_size=batch_size)
    test_features, test_labels = extract_features(test_dir, sample_count=1000, batch_size=batch_size)

    train_features = np.reshape(train_features, newshape=(2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, newshape=(1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, newshape=(1000, 4 * 4 * 512))

    # Побудова та навчання повнозв'язного класифікатора
    model = Sequential(
        [
            layers.Input(shape=(4 * 4 * 512,)),
            layers.Dense(units=256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(units=1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])
    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=batch_size,
                        validation_data=(validation_features, validation_labels))

    # Побудова грпфіків зміни втрат та точності в процесі навчання
    plot_training_history(history)
