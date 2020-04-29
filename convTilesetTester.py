import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import random

# Import images
def loadData(path):
    subdirs = [str(x) for x in Path("./" + path).iterdir() if x.is_dir()]
    x = np.empty([len(subdirs), 256, 256, 3]); y = np.empty([len(subdirs), 48, 48, 3])
    index = 0
    for i in subdirs:
        print(i)
        image = Image.open(i + '\\x.png').convert('RGB')
        data = np.asarray(image)
        x[index] = data/255

        image = Image.open(i + '\\y.png').convert('RGB')
        data = np.asarray(image)
        y[index] = data/255
        index += 1
    
    return x,y

(x_train, y_train) = loadData("tilesets")

model = load_model('tilesetmaker-to1.h5') # fewer layers
model1 = load_model('tilesetmaker-to2.h5') # fewer layers

for i in range(10):
    imgInd = random.randint(0, len(x_train)-1)
    _, axs = plt.subplots(5, 2)
    axs[0, 0].imshow(x_train[imgInd])#, cmap='gray')
    axs[1, 0].imshow(y_train[imgInd])#, cmap='gray')
    img_rgb = np.expand_dims(x_train[imgInd], axis=0)  # expand dimension
    img_rgb0 = model.predict(img_rgb)
    img_rgb1 = model1.predict(img_rgb)
    axs[0, 1].imshow(img_rgb0[0])#, cmap='gray')
    axs[1, 1].imshow(img_rgb1[0])#, cmap='gray')
    plt.show()