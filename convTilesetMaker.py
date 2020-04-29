import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation, LeakyReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
from pathlib import Path
import matplotlib.pyplot as plt
import random

# Import images
def loadData(path):
    subdirs = [str(x) for x in Path("./" + path).iterdir() if x.is_dir()]
    x = np.empty([len(subdirs)*6, 256, 256, 3]); y = np.empty([len(subdirs)*6, 48, 48, 3])
    index = 0
    for i in subdirs:
        print(i)
        imagex = Image.open(i + '\\x.png').convert('RGB')
        data = np.asarray(imagex)
        x[index] = data/255

        imagey = Image.open(i + '\\y.png').convert('RGB')
        data = np.asarray(imagey)
        y[index] = data/255
        
        data = np.asarray(ImageOps.flip(imagex))
        x[index+len(subdirs)] = data/255
        data = np.asarray(ImageOps.flip(imagey))
        y[index+len(subdirs)] = data/255
        
        data = np.asarray(ImageOps.mirror(imagex))
        x[index+2*len(subdirs)] = data/255
        data = np.asarray(ImageOps.mirror(imagey))
        y[index+2*len(subdirs)] = data/255
        
        data = np.asarray(ImageOps.flip(ImageOps.mirror(imagex)))
        x[index+3*len(subdirs)] = data/255
        data = np.asarray(ImageOps.flip(ImageOps.mirror(imagey)))
        y[index+3*len(subdirs)] = data/255
        
        data = np.asarray(ImageOps.flip(imagex.rotate(90)))
        x[index+4*len(subdirs)] = data/255
        data = np.asarray(ImageOps.flip(imagey.rotate(90)))
        y[index+4*len(subdirs)] = data/255
        
        data = np.asarray(ImageOps.flip(imagex.rotate(-90)))
        x[index+5*len(subdirs)] = data/255
        data = np.asarray(ImageOps.flip(imagey.rotate(-90)))
        y[index+5*len(subdirs)] = data/255

        index += 1
    
    # Add flipped versions of all pictures
    #for i in range(len(subdirs)):
    #    x[i+len(subdirs)] = 
    #    y[i+len(subdirs)] = 
    return x,y

(x_train, y_train) = loadData("tilesets")

input_img = Input(shape=(256, 256, 3))

# A: 256 -> 128
A = Conv2D(128, (3, 3), strides=2, padding='same')(input_img)
A = BatchNormalization(momentum=0.8)(A)
A = LeakyReLU(alpha=0.2)(A)

# B: 128 -> 64
B = Conv2D(256, (3, 3), strides=2, padding='same')(A)
B = BatchNormalization(momentum=0.8)(B)
B = LeakyReLU(alpha=0.2)(B)

# C: 64 -> 32
C = Conv2D(256, (3, 3), strides=2, padding='same')(B)
C = BatchNormalization(momentum=0.8)(C)
C = LeakyReLU(alpha=0.2)(C)

# D: 32 -> 16
D = Conv2D(512, (3, 3), strides=2, padding='same')(C)
D = BatchNormalization(momentum=0.8)(D)
D = LeakyReLU(alpha=0.2)(D)

# E: 16 -> 8
E = Conv2D(512, (3, 3), strides=2, padding='same')(D)
E = BatchNormalization(momentum=0.8)(E)
E = LeakyReLU(alpha=0.2)(E)

# F: 8 -> 4
F = Conv2D(512, (3, 3), strides=2, padding='same')(E)
F = BatchNormalization(momentum=0.8)(F)
F = LeakyReLU(alpha=0.2)(F)

# F: 4 -> 2
G = Conv2D(1024, (3, 3), strides=2, padding='same')(F)
G = BatchNormalization(momentum=0.8)(G)
G = LeakyReLU(alpha=0.2)(G)

# 2 -> 4
upscale = Conv2DTranspose(1024, (3, 3), strides=2, padding='same')(G)
upscale = BatchNormalization(momentum=0.8)(upscale)
upscale = LeakyReLU(alpha=0.2)(upscale)

# 4 -> 8
upE = concatenate([upscale, F])
upF = Conv2DTranspose(512, (3, 3), strides=2, padding='same')(upscale)
upF = BatchNormalization(momentum=0.8)(upF)
upF = LeakyReLU(alpha=0.2)(upF)

# E: 8 -> 16
upE = concatenate([upF, E])
upE = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(upE)
upE = BatchNormalization(momentum=0.8)(upE)
upE = Activation('relu')(upE)

# C: 16 -> 48
upD = concatenate([upE, D])
upD = Conv2DTranspose(3, (3, 3), strides=3, padding='same')(upD)
upD = Activation('relu')(upD)

model = Model(input_img, upD)
model.summary()

opt = Adam(beta_1=0.5)
model.compile(optimizer='adam', loss='binary_crossentropy') #0.5 beta instead of 0.9?
# fits the model on batches with real-time data augmentation:
model.fit(x_train, y_train, batch_size = 8, epochs=500) #32

for i in range(10):
    imgInd = random.randint(0, len(x_train)-1)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(x_train[imgInd])#, cmap='gray')
    img_rgb = np.expand_dims(x_train[imgInd], axis=0)  # expand dimension
    img_rgb = model.predict(img_rgb)
    axs[1].imshow(img_rgb[0])#, cmap='gray')
    plt.show()


model.save('tilesetmaker-to2.h5')