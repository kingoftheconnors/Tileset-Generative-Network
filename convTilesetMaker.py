#import argparse
import talos
import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Activation, LeakyReLU, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau#, EarlyStopping
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
from pathlib import Path
import matplotlib.pyplot as plt
import random

randomGen = np.random.default_rng()

p = {
    'filter_size': [128],
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
        print(i)
        index = randomGen.choice(indices, replace=False)
        indices.remove(index)
        imagex = Image.open(i + '\\x.png').convert('RGB')
        data = np.asarray(imagex)
        x[index] = data/255
        imagey = Image.open(i + '\\y.png').convert('RGB')
        data = np.asarray(imagey)
        y[index] = data/255

        if variate:
            index = randomGen.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey))
            y[index] = data/255
            
            index = randomGen.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.mirror(imagex))
            x[index] = data/255
            data = np.asarray(ImageOps.mirror(imagey))
            y[index] = data/255
            
            index = randomGen.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(ImageOps.mirror(imagex)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(ImageOps.mirror(imagey)))
            y[index] = data/255
            
            index = randomGen.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex.rotate(90)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey.rotate(90)))
            y[index] = data/255
            
            index = randomGen.choice(indices, replace=False)
            indices.remove(index)
            data = np.asarray(ImageOps.flip(imagex.rotate(-90)))
            x[index] = data/255
            data = np.asarray(ImageOps.flip(imagey.rotate(-90)))
            y[index] = data/255
    
    return x,y

(x_train, y_train) = loadData("tilesets")
#(x_train, y_train) = loadData("combinedTilesets")
(x_validation, y_validation) = loadData("validation_tilesets", variate=False)

def my_model(x_train, y_train, x_val, y_val, params):
    input_img = Input(shape=(256, 256, 3))
    filter_size = 128

    # A: 256 -> 128
    A = Conv2D(filter_size, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(input_img)
    A = LeakyReLU(alpha=0.2)(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = Dropout(0.2)(A)

    # B: 128 -> 64
    B = Conv2D(filter_size*2, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(A)
    B = LeakyReLU(alpha=0.2)(B)
    B = BatchNormalization(momentum=0.8)(B)
    B = Dropout(0.2)(B)

    # C: 64 -> 32
    C = Conv2D(filter_size*2, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(B)
    C = LeakyReLU(alpha=0.2)(C)
    C = BatchNormalization(momentum=0.8)(C)
    #C = Dropout(0.2)(C) #

    # D: 32 -> 16
    D = Conv2D(filter_size*4, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(C)
    D = LeakyReLU(alpha=0.2)(D)
    D = BatchNormalization(momentum=0.8)(D)
    D = Dropout(0.2)(D)

    # E: 16 -> 8
    E = Conv2D(filter_size*4, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(D)
    E = LeakyReLU(alpha=0.2)(E)
    E = BatchNormalization(momentum=0.8)(E)
    E = Dropout(0.2)(E) #

    # F: 8 -> 4
    F = Conv2D(filter_size*4, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(E)
    F = LeakyReLU(alpha=0.2)(F)
    F = BatchNormalization(momentum=0.8)(F)
    #F = Dropout(0.2)(F)

    # G: 4 -> 2
    G = Conv2D(filter_size*4, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(F)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization(momentum=0.8)(G)
    G = Dropout(0.2)(G)

    # 2 -> 4
    #upG = concatenate([upscale, G])
    upG = Conv2DTranspose(filter_size*8, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(G)
    upG = LeakyReLU(alpha=0.2)(upG)
    upG = BatchNormalization(momentum=0.8)(upG)
    #upG = Dropout(0.2)(upG) #

    # 4 -> 8
    upF = concatenate([upG, F])
    upF = Conv2DTranspose(filter_size*4, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(upF)
    upF = BatchNormalization(momentum=0.8)(upF)
    upF = Activation('relu')(upF)
    #upF = Dropout(0.2)(upF)

    # E: 8 -> 16
    upE = concatenate([upF, E])
    upE = Conv2DTranspose(filter_size*2, (3, 3), strides=2, padding='same',
        activity_regularizer=l2(1e-6))(upE)
    upE = BatchNormalization(momentum=0.8)(upE)
    upE = Activation('relu')(upE)
    #upE = Dropout(0.2)(upE) #

    # C: 16 -> 48
    upD = concatenate([upE, D])
    upD = Conv2DTranspose(3, (3, 3), strides=3, padding='same',
        activity_regularizer=l2(1e-6))(upD)
    upD = Activation('sigmoid')(upD)

    model = Model(input_img, upD)
    model.summary()

    opt = Adam(beta_1=0.5, beta_2=0.95)
    model.compile(optimizer=opt, loss='mse')#, metrics=['accuracy'])
    # fits the model on batches with real-time data augmentation:
    #es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=25)

    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size = 8,
                    epochs=200,
                    verbose = 1, callbacks=[reduce_lr])

    model.save('tilesetmaker-' + "after" + str(params['beta2']) + '.h5')

    #  "Loss"
    plt.plot(out.history['loss'])
    plt.plot(out.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.gca().set_ylim([0,1])
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("metrics/loss-after %s.png" % (str(params['beta2']*100) ))
    plt.clf()
    
    #_, axs = plt.subplots(10, 2)
    #for i in range(10):
    #    imgInd = random.randint(0, len(x_train)-1)
    #    axs[i, 0].imshow(x_train[imgInd])#, cmap='gray')
    #    img_rgb = np.expand_dims(x_train[imgInd], axis=0)  # expand dimension
    #    img_rgb = model.predict(img_rgb)
    #    axs[i, 1].imshow(img_rgb[0])#, cmap='gray')
    #plt.show()

    return out, model

talos.Scan(x_train, y_train, p, my_model, x_val=x_validation, y_val=y_validation, experiment_name="talos_output")
#my_model(x_train, y_train, x_validation, y_validation, p)

# Improvements: Test against L2 layer for if following techniques improve val-loss
# Moar data
# adding another layer of conv2D and conv2DTranspose? if that doesn't work, removing?
# Taking out lambda regularization from decoding? How about decoding?
# decrease lambda further than 1e-06
# decreasing filter size MORE
# GAN?
# ReduceLROnPlateau when val_loss plateaus to go from macro learning to fine-tuning what validation error plateaus
#               (Less patience, lower plateau factor, etc?)