import numpy as np
from keras import Input, Model
from keras.src.layers import Dense

"""
Задані два нейрони у вихідному шарі з одним входом x=4, вагами входів для кожного нейрона w1=-1, w2=-2, зміщеннями x01=-3, x02=-2, функцією активації softmax.

Визначити значення на виходах нейронів
"""

# Вхідні дані
x = 4
w1 = -1
w2 = -2
x01, x02 = -3, -2
inputs = np.array([[x]])

# Визначення моделі з використанням Keras
input_layer = Input(shape=(1,))
dense_layer = Dense(2, activation='softmax', use_bias=True,
                    kernel_initializer='ones', bias_initializer='ones')(input_layer)
model = Model(inputs=input_layer, outputs=dense_layer)

# Встановлення ваг і зміщення
model.layers[1].set_weights([np.array([[w1, w2]]), np.array([x01, x02])])

# Обчислення виходу нейрона
output = model.predict(inputs)
print("Значення на виході нейрона:", output[0])
