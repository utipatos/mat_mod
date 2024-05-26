# !pip install keras==2.9.0 tensorflow==2.9.0 matplotlib==3.8.4 scipy==1.13.0

from random import randint

from keras import models
from keras.models import load_model
import keras.utils as image
import numpy as np
import matplotlib.pyplot as plt


def display_image(img, filename):
    plt.imshow(img)
    plt.savefig(filename)


model = load_model('/content/cats_and_dogs_small_2.h1')
model.summary()

# Попередня обробка єдиного зображення
dog_index = randint(1500, 2000)
img_path = f'/content/drive/MyDrive/cats_and_dogs_small/test/dogs/dog.{dog_index}.jpg'

# Перетворення зображення на чотиривимірний тензор
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
print(img_tensor.shape)
img_tensor = np.expand_dims(img_tensor, axis=0)
print(img_tensor.shape)
img_tensor /= 255.
print(img_tensor.shape)

# Відображення вхідного зображення
plt.imshow(img_tensor[0])
plt.savefig('dog_original.png')

# Вибір виходу верхніх восьми шарів
layer_outputs = [layer.output for layer in model.layers[:8]]
# Створення моделі, яка поверне ці виходи з урахуванням заданого входу
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# ----------------------------------------------------
# Запуск моделі в режимі прогнозування
activations = activation_model.predict(img_tensor)
# Поверне список із п'ятьма масивами Numpy: по одному на кожну активацію шару
print(activations)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# ----------------------------------------------------


# Візуалізація четвертого каналу
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.savefig('dog_4_layer.png')

# Візуалізація сьомого каналу
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.savefig('dog_7_layer.png')

# Візуалізація всіх каналів для всіх проміжних активацій
# Отримати імена шарів для відображення на рисунку
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
# Цикл відображення карт ознак
for layer_name, layer_activation in zip(layer_names, activations):
    # Кількість ознак у карті ознак
    n_features = layer_activation.shape[-1]

    # Карта ознак має форму (1, size, size, n_features)
    size = layer_activation.shape[1]

    # Кількість колонок у матриці відображення каналів
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # Виведення кожного фільтра у велику горизонтальну сітку
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
