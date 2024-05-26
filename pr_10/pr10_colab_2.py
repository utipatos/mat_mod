# !pip install keras==2.9.0 tensorflow==2.9.0 matplotlib==3.8.4 scipy==1.13.0

from random import randint

from keras import models
from keras.models import load_model
import keras.utils as image
import numpy as np
import matplotlib.pyplot as plt

# Визначення тензора втрат для візуалізації фільтра
from tensorflow.keras.applications import VGG16
from keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet',
              include_top=False)
layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# Отримання градієнта втрат щодо входу моделі
# Виклик gradients повертає список тензорів (у даному разі з розміром 1). Тому зберігається лише перший елемент (тензор)
grads = K.gradients(loss, model.input)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

import numpy as np

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# Максимізація втрат стохастичним градієнтним спуском
# Початкове зображення з чорно-білим шумом
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# Величина кожної зміни градієнта
step = 1.
for i in range(40):
    # Обчислення значень втрат і градієнта
    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step
    # 40 кроків градієнтного сходження
# Коригування вхідного зображення в напрямку максимізації втрат
print(loss_value)
print(grads_value)
print(input_img_data)


# Функція перетворення тензора на допустиме зображення
def deprocess_image(x):
    # Нормалізація: виходить тензор із середнім значенням 0 і стандартним відхиленням 0,1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # Обмежує значення діапазоном [0, 1]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # Перетворить в масив значень RGB
    return x


# Функція, що генерує зображення, яке представляє фільтр
def generate_pattern(layer_name, filter_index, size=150):
    # Конструювання функції втрат, що максимізує активацію n-го фільтра в заданому шарі
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Обчислення градієнта вхідного зображення з урахуванням цих втрат
    grads = K.gradients(loss, model.input)[0]

    # Трюк з нормалізацією: нормалізує градієнт
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Повернення тензорів втрат та градієнта для даного вхідного зображення
    iterate = K.function([model.input], [loss, grads])

    # Початкове зображення з чорно-білим шумом
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 40 кроків градієнтного сходження
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))

# Створення сітки з усіма шаблонами відгуків фільтрів у шарі
layer_name = 'block1_conv1'
size = 64
margin = 5

# Чисте зображення (чорний фон) для збереження результатів
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
results = results.astype('uint8')
for i in range(8):
    # Ітерації по рядках сітки з результатами
    for j in range(8):
        # Ітерації по стовпцях сітки з результатами
        # Генерація шаблону для фільтра i+ (j*8) у шарі з іменем layer_name

        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        plt.imshow(filter_img)
        # Переписування шаблону в квадрат (i, j) всередині сітки з результатами
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
        vertical_start: vertical_end, :] = filter_img

# Відображення сітки
plt.figure(figsize=(20, 20))
plt.imshow(results)
