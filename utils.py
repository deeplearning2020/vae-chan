import tensorflow as tf
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose, 
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, 
            Reshape, GlobalAveragePooling2D, Layer)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputSpec

class conv_block(object):
    def __init__(self, filters, kernelSize, strides = 1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    def __call__(self, x, training = None):
        x = Conv2D(self.filters, self.kernelSize, kernel_initializer = init, strides = self.strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


