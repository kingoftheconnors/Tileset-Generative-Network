# author: Connor McPHerson
# references and help from:
# THIS guy who I have NO idea what he was thinking on some points:
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# DCGAN paper https://arxiv.org/pdf/1511.06434.pdf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, \
    BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, UpSampling2D, \
    Dropout, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Generator
generator = Sequential()
generator.add(Input(shape=(14, 14, 1)))

#generator.add(Dense(256 * 7 * 7, activation="relu", input_shape=(10, 10, 1)))
#generator.add(Reshape((7, 7, 256)))
generator.add(Dropout(0.4))

#generator.add(UpSampling2D())
generator.add(Conv2DTranspose(128, (5, 5), padding='same', activation='relu'))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D())
generator.add(Conv2DTranspose(64, (5, 5), padding='same', activation='relu'))

generator.add(Conv2DTranspose(32, (5, 5), padding='same', activation='relu'))
generator.add(BatchNormalization(momentum=0.8))

generator.add(Conv2DTranspose(1, (5, 5), padding='same', activation='tanh'))

generator.summary()

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, (5, 5), strides=2, input_shape=(28, 28, 1), padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))

discriminator.add(Conv2D(128, (5, 5), strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))

discriminator.add(Conv2D(256, (5, 5), strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))

discriminator.add(Conv2D(512, (5, 5), strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

# Load the dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(type(X_train), type(Y_train), type(X_test), type(Y_test))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

X_test = X_test / 127.5 - 1.
X_test = np.expand_dims(X_test, axis=3)

# Adversarial ground truths
valid = np.ones((16, 1))
fake = np.zeros((16, 1))

z = Input(shape=(14, 14, 1))
img = generator(z)
discriminator.trainable = False

validity = discriminator(img)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')

# Method of loading images from https://blog.tanka.la/2018/10/28/build-the-mnist-model-with-your-own-handwritten-digits-using-tensorflow-keras-and-python/
def loadImage(x):
    idy = np.random.randint(0, X_test.shape[0])
    while Y_test[idy] != x:
        idy = np.random.randint(0, X_test.shape[0])
    img = X_test[idy]
    img = np.resize( img, (14,14))
    img = Image.fromarray(img)
    img.resize(size=(14, 14))
    img = np.array(img)
    img = np.resize( img, (14,14,1))
    return img

for epoch in range(10001):

    idx = np.random.randint(0, X_train.shape[0], 16)
    real_imgs = X_train[idx]
    in_imgs = np.array([Y_train[idx]])

    in_imgs = np.swapaxes(in_imgs, 0, 1)
    in_imgs = np.apply_along_axis(loadImage, 1, in_imgs)

    gen_imgs = generator.predict(in_imgs)

    d_loss_real = discriminator.train_on_batch(real_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_acc = round(float(d_loss[1]), 4)
    d_loss = round(float(d_loss[0]), 4)

    idx = np.random.randint(0, X_train.shape[0], 16)
    in_imgs = np.array([Y_train[idx]])
    in_imgs = np.swapaxes(in_imgs, 0, 1)
    in_imgs = np.apply_along_axis(loadImage, 1, in_imgs)

    g_loss = round(float(combined.train_on_batch(in_imgs, valid)), 4)

    comb_str = "Epoch {:.5s}: {:.40s} | {} ".format(f"{epoch}{' ' * 100}",
                                                    f"Disciminator - Loss: {d_loss}, Acc: {d_acc} {' ' * 100}",
                                                    f"Generator - Loss: {g_loss}")
    print(comb_str)

    if epoch % 200 == 0:
        r, c = 5, 5
        
        idx = np.random.randint(0, X_train.shape[0], 25)
        in_imgs = np.array([Y_train[idx]])
        in_imgs = np.swapaxes(in_imgs, 0, 1)
        in_imgs = np.apply_along_axis(loadImage, 1, in_imgs)
        
        gen_imgs = generator.predict(in_imgs)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("imagesmlp/%d.png" % epoch)
        plt.close()