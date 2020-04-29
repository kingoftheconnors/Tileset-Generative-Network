import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
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

input_img = Input(shape=(256, 256, 3))
x = Conv2D(256, (3, 3), activation='relu')(input_img)
x = Conv2D(256, (3, 3), activation='relu')(input_img)
x = BatchNormalization(momentum=0.7)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu')(x)
x = Conv2D(512, (5, 5), activation='relu')(x)
x = BatchNormalization(momentum=0.7)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, (3, 3), activation='relu')(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = Conv2D(512, (5, 5), activation='relu')(x)
x = Conv2D(3, (5, 5), activation='relu')(x)

model = Model(input_img, x)
model.summary()

model.compile(optimizer='adadelta', loss='binary_crossentropy') #adam?

print(len(x_train), len(y_train))
callback = EarlyStopping(monitor='val_loss', patience=10)
# fits the model on batches with real-time data augmentation:
model.fit(x_train, y_train, batch_size = 4, epochs=500, callbacks = [callback]) #32

for i in range(10):
    imgInd = random.randint(0, len(x_train)/4-1)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(x_train[imgInd])#, cmap='gray')
    img_rgb = np.expand_dims(x_train[imgInd], axis=0)  # expand dimension
    img_rgb = model.predict(img_rgb)
    axs[1].imshow(img_rgb[0])#, cmap='gray')
    plt.show()


model.save('tilesetmakerOld(without sigmoid).h5')