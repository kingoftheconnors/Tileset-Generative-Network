#import argparse
import talos
import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation, LeakyReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
from pathlib import Path
import matplotlib.pyplot as plt

random = np.random.default_rng()

#parser = argparse.ArgumentParser(description="Tileset Creator Model Maker")
#parser.add_argument('--version', type=str, default='')
#args = parser.parse_args()

version = "test"
dropout = 0.2

p = {
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
    'optimizer': ['adam', 'adam2', 'adagrad', 'adadelta', 'RMSprop'],
    'batch_size': [2, 4, 8, 16],
    'epoch': [50, 100, 200]
}

# Import images
def loadData(path, variate=True):
    subdirs = [str(x) for x in Path("./" + path).iterdir() if x.is_dir()]
    size = len(subdirs)
    if variate:
        size *= 6
    
    x = np.empty([size, 256, 256, 3]); y = np.empty([size, 48, 48, 3])
    indices = [i for i in range(size)]
    for i in subdirs:
        index = random.choice(indices, replace=False)
        indices.remove(index)
        imagex = Image.open(i + '\\x.png').convert('RGB')
        data = np.asarray(imagex)
        x[index] = data/255
        imagey = Image.open(i + '\\y.png').convert('RGB')
        data = np.asarray(imagey)
        y[index] = data/255

        if variate:
            index = random.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey))
            y[index] = data/255
            
            index = random.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.mirror(imagex))
            x[index] = data/255
            data = np.asarray(ImageOps.mirror(imagey))
            y[index] = data/255
            
            index = random.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(ImageOps.mirror(imagex)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(ImageOps.mirror(imagey)))
            y[index] = data/255
            
            index = random.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex.rotate(90)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey.rotate(90)))
            y[index] = data/255
            
            index = random.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex.rotate(-90)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey.rotate(-90)))
            y[index] = data/255
    
    return x,y

(x_train, y_train) = loadData("tilesets")
(x_val, y_val) = loadData("validation_tilesets", variate=False)

def my_model(x_train, y_train, x_val, y_val, params):
    input_img = Input(shape=(256, 256, 3))

    # A: 256 -> 128
    A = Conv2D(128, (3, 3), strides=2, padding='same')(input_img)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    A = Dropout(0.1)(A)

    # B: 128 -> 64
    B = Conv2D(128*2, (3, 3), strides=2, padding='same')(A)
    B = BatchNormalization(momentum=0.8)(B)
    B = LeakyReLU(alpha=0.2)(B)
    B = Dropout(0.1)(B)

    # C: 64 -> 32
    C = Conv2D(128*2, (3, 3), strides=2, padding='same')(B)
    C = BatchNormalization(momentum=0.8)(C)
    C = LeakyReLU(alpha=0.2)(C)
    C = Dropout(0.1)(C)

    # D: 32 -> 16
    D = Conv2D(128*4, (3, 3), strides=2, padding='same')(C)
    D = BatchNormalization(momentum=0.8)(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Dropout(0.1)(D)

    # E: 16 -> 8
    E = Conv2D(128*4, (3, 3), strides=2, padding='same')(D)
    E = BatchNormalization(momentum=0.8)(E)
    E = LeakyReLU(alpha=0.2)(E)
    E = Dropout(0.1)(E)

    # F: 8 -> 4
    F = Conv2D(128*4, (3, 3), strides=2, padding='same')(E)
    F = BatchNormalization(momentum=0.8)(F)
    F = LeakyReLU(alpha=0.2)(F)
    F = Dropout(0.1)(F)

    # G: 4 -> 2
    G = Conv2D(128*4, (3, 3), strides=2, padding='same')(F)
    G = BatchNormalization(momentum=0.8)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = Dropout(0.1)(G)

    # 2 -> 4
    #upG = concatenate([upscale, G])
    upG = Conv2DTranspose(128*8, (3, 3), strides=2, padding='same')(G)
    upG = BatchNormalization(momentum=0.8)(upG)
    upG = LeakyReLU(alpha=0.2)(upG)
    upG = Dropout(0.1)(upG)

    # 4 -> 8
    upF = concatenate([upG, F])
    upF = Conv2DTranspose(128*4, (3, 3), strides=2, padding='same')(upF)
    upF = BatchNormalization(momentum=0.8)(upF)
    upF = LeakyReLU(alpha=0.2)(upF)
    upF = Dropout(0.1)(upF)

    # E: 8 -> 16
    upE = concatenate([upF, E])
    upE = Conv2DTranspose(128*2, (3, 3), strides=2, padding='same')(upE)
    upE = BatchNormalization(momentum=0.8)(upE)
    upE = Activation('relu')(upE)
    upE = Dropout(0.1)(upE)

    # C: 16 -> 48
    upD = concatenate([upE, D])
    upD = Conv2DTranspose(3, (3, 3), strides=3, padding='same')(upD)
    upD = Activation('sigmoid')(upD)

    model = Model(input_img, upD)
    model.summary()

    opt = Adam(beta_1=0.5)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #0.5 beta instead of 0.9?
    # fits the model on batches with real-time data augmentation:
    #es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size = 8,
                    epochs=200,
                    verbose = 0)#, callbacks=[es]) #32

    return out, model

##  "Accuracy"
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig("metrics/acc-%s.png" % version)
#plt.clf()
## "Loss"
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig("metrics/loss-%s.png" % version)
#plt.clf()

#for i in range(5):
#    imgInd = random.randint(0, len(x_train)-1)
#    _, axs = plt.subplots(1, 2)
#    axs[0].imshow(x_train[imgInd])#, cmap='gray')
#    img_rgb = np.expand_dims(x_train[imgInd], axis=0)  # expand dimension
#    img_rgb = model.predict(img_rgb)
#    axs[1].imshow(img_rgb[0])#, cmap='gray')
#    plt.show()

talos.Scan(x_train, y_train, p, my_model, x_val=x_test, y_val=y_test, experiment_name="talos_output")
#model.save('tilesetmaker-' + version + '.h5')

# Tests:
# Batch Sizes
# L2 limits
# Epoch Number (50, 100, or 200?)
#
# GAN?