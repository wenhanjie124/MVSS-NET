# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow.keras.utils import plot_model
from tools.utils import *


class MVCNN:
    def __init__(self, input_shape=None, classes=1000):
        self.input_shape = input_shape
        self.classes = classes
        return

    def build(self):
        inputs = layers.Input(shape=self.input_shape, name="input")
        # x = tf.split(value=inputs, num_or_size_splits=12, axis=1)
        x = layers.Lambda(self.squeeze_tensor, name="split")(inputs)
        # extract_model = self.extract_layer(name="extract_layer_")
        extract_model = keras.applications.VGG16(include_top=False,
                                                 weights='data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                                 input_tensor=None,
                                                 input_shape=[227, 227, 3],
                                                 pooling=None)
        extract_model.trainable = True
        plot_model(extract_model, "fuck.png", show_shapes=True)
        view_pool = [extract_model(i) for i in x]
        # x = layers.Concatenate(name="concatenate_layer")(view_pool)
        x = layers.Maximum(name="max_layer")(view_pool)
        x = layers.GlobalAveragePooling2D(name="golbal")(x)
        x = layers.Dense(4096, activation="relu", name="fc_layer1")(x)
        x = layers.Dense(2048, activation="relu", name="fc_layer2")(x)
        prediction = layers.Dense(self.classes, activation="softmax", name="prediction")(x)
        return keras.models.Model(inputs, prediction, name='MVCNN')

    def squeeze_tensor(self, inputs):
        slices = []
        for i in range(0, 12):
            slices.append(inputs[:, i, :, :, :])
        return slices

    # def squeeze_tensor(self, inputs):
    #     slices = [tf.squeeze(i, axis=1) for i in inputs]
    #     return slices
    def extract_layer(self, name):
        input_shape2 = self.input_shape[-3:]
        inputs = keras.Input(shape=input_shape2, name=name + "input")
        x = layers.Conv2D(32, 7, 2, padding="same", name=name + "_conv1")(inputs)
        x = layers.BatchNormalization(name=name + "_conv1_bn", epsilon=1.001e-5)(x)
        x = layers.Activation("relu", name=name + "_conv1_relu1")(x)
        x = layers.MaxPooling2D(3, 2, name=name + "maxpool")(x)
        x = self.senet(x, 32, name=name + "resblock0")
        x = self.senet(x, 32, name=name + "resblock1")
        x = self.senet(x, 64, name=name + "resblock2")
        x = self.senet(x, 64, name=name + "resblock3")
        x = self.senet(x, 128, name=name + "resblock4")
        x = self.senet(x, 128, name=name + "resblock5")
        extract = layers.GlobalAveragePooling2D(name=name + "globalaverpool")(x)
        # extract = x
        return keras.Model(inputs=inputs, outputs=extract, name=name)

    def senet(self, inputs, fil, name):
        x = layers.Conv2D(fil, 3, 1, padding="same",
                          name=name + "_conv1")(inputs)
        x = layers.BatchNormalization(name=name + "_bn1", epsilon=1.001e-5)(x)
        x = layers.Activation("relu", name=name + "_relu1")(x)

        x = layers.Conv2D(fil, 3, 1, padding="same",
                          name=name + "_conv2")(x)
        x = layers.BatchNormalization(name=name + "_bn2", epsilon=1.001e-5)(x)
        """
        SEnet 内容
        """
        # x = layers.Activation("relu", name=name + "_relu2")(x)
        # x1 = layers.GlobalAveragePooling2D()(x)
        # x1 = layers.Reshape((1, 1, keras.backend.int_shape(x1)[-1]), name=name + "_atten_reshape")(x1)
        # x2 = layers.Dense(keras.backend.int_shape(x1)[-1] // 16, name=name + "_atten_1_1_c16",
        #                   activation="relu")(x1)
        # x1 = layers.Dense(keras.backend.int_shape(x1)[-1], name=name + "_atten_1_1_c", activation="sigmoid")(x2)
        # x = layers.Multiply(name=name + "_atten_multiply")([x, x1])

        short_cut = layers.Conv2D(fil, 3, 1, padding="same",
                                  name=name + "_input_conv")(inputs)
        short_cut = layers.BatchNormalization(name=name + "_input_bn", epsilon=1.001e-5)(short_cut)
        x = layers.Add()([x, short_cut])
        x = layers.Activation("relu", name=name + "_relu3")(x)
        x = layers.MaxPooling2D(2, 2, padding="same", name=name + "_maxpool")(x)
        return x
