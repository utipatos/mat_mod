from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
import matplotlib.pyplot as plt

from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.utils as image
import numpy as np
img_path = '/content/drive/MyDrive/creative_commons_elephant.jpg'

# Зображення 224 × 224 у форматі бібліотеки Python Imaging Library (PIL)
img=image.load_img(img_path)
plt.matshow(img)

img=image.load_img(img_path, target_size=(224, 224))
# Масив Numpy з числами типу float32, що має форму (224, 224, 3)
x=image.img_to_array(img)

# Додавання розмірності для перетворення масиву в пакет з формою (1, 224, 224, 3)
x= np.expand_dims(x, axis=0)

# Попередня обробка пакета (нормалізація каналів кольору)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
np.argmax(preds[0])

# Реалізація алгоритму Grad-CAM
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import sys
sys.setrecursionlimit(100000)

#Елемент «африканський слон» в векторі прогнозів
african_elephant_output = model.output[:, 386]

#Вихідна карта ознак шару block5_conv3, останнього згорткового шару в мережі VGG16
last_conv_layer = model.get_layer('block5_conv3')

# Градієнт класу «африканський слон» для вихідної карти ознак шару block5_conv3
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

#Вектор з формою (512,), кожен елемент якого визначає інтенсивність градієнта для заданого каналу в карті ознак
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
#Дозволяє отримати доступ до значень тільки що визначених величин: pooled_grads і вихідної карти ознак шару block5_conv3 для заданого зображення
[pooled_grads, last_conv_layer. output[0]])

# Значення цих двох величин у вигляді масивів Numpy для даного зразка зображення двох слонів
pooled_grads_value, conv_layer_output_value = iterate([x])

# Помножує кожен канал у карті ознак на «важливість цього каналу» для класу «слон»
for i in range(512):
  conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Середнє для каналів в отриманій карті ознак — це теплова карта активації класу
heatmap =np.mean(conv_layer_output_value, axis=-1)

# Заключна обробка теплової карти
import matplotlib.pyplot as plt
import numpy as np
heatmap = np.maximum(heatmap, 0)
#heatmap / = np.max(heatmap)
plt.matshow(heatmap)

import cv2
# Завантаження вхідного зображення за допомогою cv2
img = cv2.imread(img_path)
#Зміна розмірів теплової карти у відповідності з розмірами оригінальної фотографії
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
plt.matshow(heatmap)
