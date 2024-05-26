import numpy as np
from keras import Input, Model
from keras.src.activations import elu
from keras.src.layers import Dense

"""
Заданий двовходовий нейрон з входами x1=0.2, x2=0.2,
вагами відповідних входів w1=0.3, w2=0.3, зміщенням x0=0.3, функція активації ‘elu’ (a=1).
Визначити значення на виході нейрона.
"""

# Вхідні дані
x1 = 0.2
x2 = 0.2
inputs = np.array([[x1, x2]])

# Визначення моделі з використанням Keras
input_layer = Input(shape=(2,))
dense_layer = Dense(1, activation=lambda x: elu(x, alpha=1), use_bias=True,
                    kernel_initializer='ones', bias_initializer='ones')(input_layer)
model = Model(inputs=input_layer, outputs=dense_layer)

# Встановлення ваг і зміщення
model.layers[1].set_weights([np.array([[0.3], [0.3]]), np.array([0.3])])

# Обчислення виходу нейрона
output = model.predict(inputs)
print("Значення на виході нейрона:", output[0][0])
