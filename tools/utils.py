import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import os
import sys


def dataset_to_numpy(dataset, print_shape=False):
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset):
        images.append(image.numpy())
        labels.append(label.numpy())
    images = np.asarray(images)
    labels = np.asarray(labels)
    if print_shape:
        print(images.shape)
        print(labels.shape)
    return images, labels


def cv_show_image(img_name, img_dict):
    '''
    显示图片
    :param img_name: 图片窗口名字
    :param img_dict: 图片矩阵
    :return:
    '''
    cv2.namedWindow(img_name, 0)
    cv2.imshow(img_name, img_dict)
    cv2.waitKey(0)
    return


def plt_show_image(title, image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()
    plt.close()
    return
