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
def dataset_generator(dataset, batch_size, len, left, right):
    images = []
    labels = []
    tot = 0
    while 1:
        for k, (image, label) in enumerate(dataset):
            if (label.numpy() >= left) and (label.numpy() <= right):
                images.append(image.numpy())
                labels.append(label.numpy() - left)
                tot += 1
                if tot == batch_size:
                    yield np.asarray(images), np.asarray(labels)
                    tot = 0
                    images = []
                    labels = []


if __name__ == "__main__":
    if not os.path.exists(f"modelcheck"):
        os.makedirs(f"modelcheck")
    modelcheck_name = f"modelcheck"
    # ran = random.randint(80, 130)
    ll = 0
    rr = 299
    print("*" * 20 + f'll=={ll}    rr=={rr}' + "*" * 20)
    train_data = Read_mul_view_tfrecords('data/人工分型_total_list_500.tfrecords', gray=True)
    # train_len = Get_tfrecords_nums('data/total.tfrecords')
    train_len = 0
    for j, (image, label) in enumerate(train_data):
        if (label.numpy() >= ll) and (label.numpy() <= rr):
            train_len += 1
    print("*" * 20 + f"模型数量为：{train_len}" + "*" * 20)
    model = MVCNN(input_shape=(12, 500, 500, 1), classes=300).build()
    plot_model(model, f'{modelcheck_name}\\mvcnn.png', show_shapes=True)
    model.compile(optimizer=SGD(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
    modelcheck = ModelCheckpoint(
        filepath=f"{modelcheck_name}\\" + 'mvcnn_epoc{epoch:04d}loss{loss:.4f}acc{sparse_categorical_accuracy:.4f}topk{sparse_top_k_categorical_accuracy}.hdf5',
        monitor='loss', verbose=1, save_weights_only=True, save_best_only=True,
        period=1)
    easystop = EarlyStopping(monitor='loss', min_delta=0.001, patience=100,
                             verbose=1, mode='auto')
    lr_auto = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                patience=20,
                                verbose=1, mode='auto',
                                min_delta=0.001, cooldown=0, min_lr=1e-8)
    tensorboard = TensorBoard(log_dir=f'{modelcheck_name}' + '\\logs', histogram_freq=0,
                              write_graph=True,
                              write_grads=False, write_images=True, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')
    model.fit_generator(dataset_generator(train_data, batch_size=2, len=train_len, left=ll, right=rr),
                        steps_per_epoch=train_len // 2,
                        verbose=1, max_queue_size=5, epochs=1000,
                        callbacks=[easystop, lr_auto, modelcheck, tensorboard])
    with open(f'{modelcheck_name}\\left_right.txt', "a+") as f:
        f.write("num=" + f"{rr - ll + 1}" + f"   l==={ll} r==={rr}")
    print('**' * 20 + '\n')
    print(f"训练结束！")
    print("**" * 20 + '\n')
    shutil.copy("tools/MVCNN.py", "modelcheck/MVCNN.py")
    shutil.move("fuck.png", "modelcheck/fuck.png")
    shutil.copy("start_train.py", "modelcheck/start_train.py")
    print("完成剩余操作")
