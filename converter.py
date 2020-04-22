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
import matplotlib.pyplot as plt


# Generator
generator = Sequential()
generator.add(Dense(256 * 7 * 7, activation="relu", input_dim=100))
generator.add(Reshape((7, 7, 256)))
generator.add(Dropout(0.4))

generator.add(UpSampling2D())
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
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

# Adversarial ground truths
valid = np.ones((16, 1))
fake = np.zeros((16, 1))

z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False

validity = discriminator(img)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')

for epoch in range(10001):
    idx = np.random.randint(0, X_train.shape[0], 16)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (16, 100))

    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_acc = round(float(d_loss[1]), 4)
    d_loss = round(float(d_loss[0]), 4)

    noise = np.random.normal(0, 1, (16, 100))
    g_loss = round(float(combined.train_on_batch(noise, valid)), 4)

    comb_str = "Epoch {:.5s}: {:.40s} | {} ".format(f"{epoch}{' ' * 100}",
                                                    f"Disciminator - Loss: {d_loss}, Acc: {d_acc} {' ' * 100}",
                                                    f"Generator - Loss: {g_loss}")
    print(comb_str)

    if epoch % 200 == 0:
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(noise)

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