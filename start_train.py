# -*- coding: utf-8 -*-
import shutil
from tensorflow.python.keras.utils.vis_utils import plot_model
from create_mul_view_record import Read_mul_view_tfrecords
from tools.MVCNN import MVCNN
from tools.utils import *
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.optimizers import SGD
import numpy as np

GPU = "0"
os.environ["PATH"] += os.pathsep + 'D:/Program Files/graphviz-2.38/release/bin'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU


# 73

if __name__ == "__main__":
    if not os.path.exists(f"modelcheck"):
        os.makedirs(f"modelcheck")
    modelcheck_name = f"modelcheck"
    # ran = random.randint(80, 130)
    print("*" * 20 + f'll=={ll}    rr=={rr}' + "*" * 20)
    train_data = Read_mul_view_tfrecords('data/人工分型_total_list_500.tfrecords', gray=True)
    # train_len = Get_tfrecords_nums('data/total.tfrecords')
    train_len = 0
    for j, (image, label) in enumerate(train_data):
        if (label.numpy() >= ll) and (label.numpy() <= rr):
            train_len += 1
    print("*" * 20 + f"{train_len}" + "*" * 20)
    model = MVCNN(input_shape=(12, 500, 500, 1), classes=300).build()
    plot_model(model, f'{modelcheck_name}\\mvcnn.png', show_shapes=True)
    model.compile(optimizer=SGD(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
    model.fit_generator(dataset_generator(train_data, batch_size=2, len=train_len, left=ll, right=rr),
                        steps_per_epoch=train_len // 2,
                        verbose=1, max_queue_size=5, epochs=1000)

