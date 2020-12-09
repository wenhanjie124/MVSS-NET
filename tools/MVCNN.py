# -*- coding: utf-8 -*-
from tensorflow.python.keras.layers import Input, BatchNormalization, Activation, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers.merge import Concatenate, Add
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model


class MVCNN:
    def __init__(self, input_shape=None, classes=1000):
        self.input_shape = input_shape
        self.classes = classes

    def build(self):
        inputs = Input(shape=self.input_shape, name="input")
        # x = tf.split(value=inputs, num_or_size_splits=12, axis=1)
        x = Lambda(self.squeeze_tensor, name="split")(inputs)
        extract_model = self.extract_layer(name="extract_layer_")
        plot_model(extract_model, "fuck.png", show_shapes=True)
        view_pool = [extract_model(i) for i in x]
        x = Concatenate(name="concatenate_layer")(view_pool)
        # x = layers.Maximum()(view_pool)
        x = Dense(2048, activation="relu", name="fc_layer1")(x)
        # # x = layers.BatchNormalization(name="fc_batch", epsilon=1.001e-5)(x)
        x = Dense(1024, activation="relu", name="fc_layer2")(x)
        # x = layers.GlobalAveragePooling2D(name="globalaverpool")(x)
        prediction = Dense(self.classes, activation="softmax", name="prediction")(x)
        return Model(inputs, prediction, name='MVCNN')

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
        inputs = Input(shape=input_shape2, name=name + "input")
        x = Conv2D(64, 7, 2, padding="same", name=name + "_conv1")(inputs)
        x = BatchNormalization(name=name + "_conv1_bn")(x)
        x = Activation("relu", name=name + "_conv1_relu1")(x)
        x = MaxPooling2D(3, 2, name=name + "maxpool")(x)
        x = self.senet(x, 64, name=name + "resblock0")
        x = self.senet(x, 64, name=name + "resblock1")
        x = self.senet(x, 128, name=name + "resblock2")
        x = self.senet(x, 128, name=name + "resblock3")
        x = self.senet(x, 256, name=name + "resblock4")
        x = self.senet(x, 256, name=name + "resblock5")
        extract = GlobalAveragePooling2D(name=name + "globalaverpool")(x)
        # extract = x
        return Model(inputs=inputs, outputs=extract, name=name)

    def senet(self, inputs, fil, name):
        x = Conv2D(fil, 3, 1, padding="same",
                   name=name + "_conv1")(inputs)
        x = BatchNormalization(name=name + "_bn1")(x)
        x = Activation("relu", name=name + "_relu1")(x)

        x = Conv2D(fil, 3, 1, padding="same",
                   name=name + "_conv2")(x)
        x = BatchNormalization(name=name + "_bn2")(x)
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

        short_cut = Conv2D(fil, 3, 1, padding="same",
                           name=name + "_input_conv")(inputs)
        short_cut = BatchNormalization(name=name + "_input_bn")(short_cut)
        x = Add()([x, short_cut])
        x = Activation("relu", name=name + "_relu3")(x)
        x = MaxPooling2D(2, 2, padding="same", name=name + "_maxpool")(x)
        return x
