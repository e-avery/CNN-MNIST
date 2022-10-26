# convolutional neural network for image classification - MNIST dataset
import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Input,Dense,Conv2D,BatchNormalization,Dropout,Flatten,LeakyReLU,Activation

# load the dataset - 50,000 28x28 pixel grayscale images w/ 10,000 extra used for testing/evaluation
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# given that each image is grayscale, normalize to avoid an exploding gradient during training
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# resize 4D format (batch,rows,cols,channels)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test  = x_test.reshape(10000, 28, 28, 1)

# labels 0-9
num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test,  num_classes)

# each image size with a single channel
input_layer = Input((28,28,1))

# normalization was performed manually before input, so pass directly to first convolutional layer
x = Conv2D(
    filters = 10,
    kernel_size = (3,3),
    strides = 1,
    padding = 'same'    
    )(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(
    filters = 20,
    kernel_size = 3,
    strides = 2,
    padding = 'same'
    )(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# flatten before handing to dense layers
x = Flatten()(x)

x = Dense(units = 200)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Dense(units = 150)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# drop 50% during training for regularization
x = Dropout(rate = 0.5)(x)

x = Dense(num_classes)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer,output_layer)

# observe the model output
model.summary()

# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 28, 28, 1)         0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 28, 28, 10)        20
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 28, 28, 10)        40
# _________________________________________________________________
# leaky_re_lu_1 (LeakyReLU)    (None, 28, 28, 10)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 14, 14, 20)        1820
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 14, 14, 20)        80
# _________________________________________________________________
# leaky_re_lu_2 (LeakyReLU)    (None, 14, 14, 20)        0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 3920)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 200)               784200
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 200)               800
# _________________________________________________________________
# leaky_re_lu_3 (LeakyReLU)    (None, 200)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 150)               30150
# _________________________________________________________________
# batch_normalization_4 (Batch (None, 150)               600
# _________________________________________________________________
# leaky_re_lu_4 (LeakyReLU)    (None, 150)               0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 150)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1510
# _________________________________________________________________
# activation_1 (Activation)    (None, 10)                0
# =================================================================
# Total params: 819,220
# Trainable params: 818,460
# Non-trainable params: 760
# _________________________________________________________________

# assign optimizer, compile, and train
opt = Adam(learning_rate = 0.0005)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size = 28, epochs = 10, shuffle = True)

# labels for classification/evaluation
classes = [str(_) for _ in range(num_classes)]

# model evaluation
model.evaluate(x_test, y_test, batch_size = 1000)

# object returned from .fit()
#hist.params
#hist.epoch
#hist.history
