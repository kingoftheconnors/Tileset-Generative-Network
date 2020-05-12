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

model2 = load_model('tilesetmaker-before.h5')
#model3 = load_model('tilesetmaker-before5e-06.h5')
#model4 = load_model('tilesetmaker-before1e-06.h5')

model6 = load_model('tilesetmaker-after32.h5')
model7 = load_model('tilesetmaker-after64.h5')
model8 = load_model('tilesetmaker-after128.h5')
model9 = load_model('tilesetmaker-after256.h5')

def testImage(filenames):
    _, axs = plt.subplots(len(filenames), 7)
    for i in range(len(filenames)):
        filename = filenames[i]
        image = Image.open(filename).convert('RGB')
        data = np.asarray(image)
        data = data/255
        output2 = model2.predict(np.expand_dims(data, axis=0))
        #output3 = model3.predict(np.expand_dims(data, axis=0))
        #output4 = model4.predict(np.expand_dims(data, axis=0))
        #output5 = model5.predict(np.expand_dims(data, axis=0))
        output6 = model6.predict(np.expand_dims(data, axis=0))
        output7 = model7.predict(np.expand_dims(data, axis=0))
        output8 = model8.predict(np.expand_dims(data, axis=0))
        output9 = model9.predict(np.expand_dims(data, axis=0))
        
        axs[i, 0].imshow(data)
        axs[i, 1].imshow(output2[0])
        axs[i, 3].imshow(output6[0])
        axs[i, 4].imshow(output7[0])
        axs[i, 5].imshow(output8[0])
        axs[i, 6].imshow(output9[0])
    plt.show()


subdirs = [str(x) for x in Path("./tilesets").iterdir() if x.is_dir()]
testImage(['celestefootage.png', 'pictures/wood.png', 'pictures/moonA.jpg', random.choice(subdirs) + '/x.png'])