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

model = load_model('tilesetmaker5.h5') # extra layers

def testImage(filename):
    image = Image.open(filename).convert('RGB')
    data = np.asarray(image)
    data = data/255
    output = model.predict(np.expand_dims(data, axis=0))
    _, axs = plt.subplots(2)
    axs[0].imshow(data)
    axs[1].imshow(output[0])
    plt.show()

testImage('pictures/wood.png')
testImage('celestefootage.png')
testImage('pictures/moonA.jpg')
testImage('pictures/moonB.jpg')