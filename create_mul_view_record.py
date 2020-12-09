# -*-coding: utf-8 -*-
'''
生产TFrecord文件
保存在data文件下面，有train，val，test,label文件。
'''
import random
from tools.utils import *
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def decode_image(image, gray=False):
    '''
    解码图片，返回resize后的图片
    :param image:
    :return:
    '''
    tf_height = image['height']
    tf_width = image['width']
    tf_depth = image['depth']
    tf_label = tf.cast(image['label'], tf.float32)
    tf_image = tf.io.decode_raw(image['image_raw'], tf.uint8)
    tf_image = tf.reshape(tf_image, [tf_height, tf_width, tf_depth])
    if gray:
        tf_image = tf_image[:, :, 0]
        tf_image = tf.reshape(tf_image, [tf_height, tf_width, 1])
    else:
        pass
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image = tf_image / 255.0
    return tf_image, tf_label


def decode_mul_view(image, gray=False):
    '''
    解码图片，返回resize后的图片
    :param image:
    :return:
    '''
    tf_batch = image['mul_view_b']
    tf_height = image['mul_view_h']
    tf_width = image['mul_view_w']
    tf_depth = image['mul_view_d']
    tf_label = tf.cast(image['label'], tf.int32)
    tf_image = tf.io.decode_raw(image['mul_view_raw'], tf.uint8)
    tf_image = tf.reshape(tf_image, [tf_batch, tf_height, tf_width, tf_depth])
    if gray:
        tf_image = tf_image[:, :, :, 0]
        tf_image = tf.reshape(tf_image, [tf_batch, tf_height, tf_width, 1])
    else:
        pass
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image = tf_image / 255.0
    return tf_image, tf_label


def decode_pcl(pcl):
    pcl_points = pcl['pcl_points']
    pcl_label = tf.cast(pcl['label'], tf.int32)
    pcl_pcl = tf.io.decode_raw(pcl['pcl_raw'], tf.float32)
    pcl_pcl = tf.reshape(pcl_pcl, [pcl_points, 3])
    return pcl_pcl, pcl_label


def Read_tfrecords(filepath, gray=False):
    '''
    读取TF文件，并且回复
    :param filename:tf文件路径
    :param type:返回图像的类型
           None从uint8[0-255]到float[0-255]
           normalization从uint8[0-255]归一化到float[0-1]
           cennormalization从uint8[0-255]归一化到float[0-1]并且减去均值中心化
    :return:返回图片和标签
    '''
    dataset = tf.data.TFRecordDataset(filepath)
    features_dict = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.float32)}
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, features_dict), num_parallel_calls=6)
    dataset = dataset.map(lambda x: decode_image(x, gray))
    return dataset


def Read_mul_view_tfrecords(filepath, gray=False):
    '''
    读取TF文件，并且返回
    :param filename:tf文件路径
    :return:返回点云和标签
    '''
    dataset = tf.data.TFRecordDataset(filepath)
    features_dict = {
        'mul_view_raw': tf.io.FixedLenFeature([], tf.string),
        'mul_view_b': tf.io.FixedLenFeature([], tf.int64),
        'mul_view_h': tf.io.FixedLenFeature([], tf.int64),
        'mul_view_w': tf.io.FixedLenFeature([], tf.int64),
        'mul_view_d': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)}
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, features_dict), num_parallel_calls=6)
    dataset = dataset.map(lambda x: decode_mul_view(x, gray))
    return dataset


def Read_image(image_path, resize_height, resize_width):
    '''
    :param image_path: 图片路劲
    :param resize_height: 图片重设高度
    :param resize_width: 图片重设宽度
    :param normalization: 是否归一化到0-1
    :return: 返回一个RBG的三通道数据
    '''
    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:  # 灰度转换三通道
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if resize_width > 0 and resize_height > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
    rgb_image = np.asanyarray(rgb_image)
    return rgb_image


def Read_mul_view(image_list):
    '''

    :param image_list: image_list路径
    :return:返回一个image_numpy
    '''
    image = []
    for i in image_list:
        ima = cv2.imread(i, flags=cv2.COLOR_BGR2RGB)
        ima = cv2.resize(src=ima, dsize=(500, 500), interpolation=cv2.INTER_AREA)
        image.append(ima)
    image_np = np.asanyarray(image)
    return image_np


def Get_tfrecords_nums(tfrecord_path):
    '''
    统计TFrecord文件图像的个数
    :param tfrecord_path:tf文件路径
    :return:无
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset.__iter__()
    nums = 0
    for i in dataset.__iter__():
        nums += 1
    return nums


def Load_image_label_txt(image_txt, num, shuffle):
    '''
    载入图片与标签文件，格式：图片路径 标签数目
    :param image_txt: 标签图片txt路径
    :param num: 标签数目
    :param shuffle: 是否随机初始化
    :return:图片list与标签list
    '''
    image_list = []
    label_list = []
    image = []
    with open(image_txt, 'r+') as data:
        lines_list = data.readlines()
    random.seed()
    num = 0
    num2 = 0
    for lines in lines_list:
        line = lines.rstrip().split()
        if num2 < 11:
            image.append(line[0])
            num2 += 1
        else:
            image.append(line[0])
            num2 = 0
            image_list.append(image)
            image = []
            label_list.append(line[1])
    return image_list, label_list


def Create_tfrecord(pcl_num, image_txt, image_tfrecord_output, shuffle, log):
    '''

    :param pcl_num: pcl的数量
    :param image_txt: txt文件路径
    :param image_tfrecord_output: tf文件输出路径
    :param shuffle: 是否随机化
    :param log: 是否打印
    :return:
    '''
    images_list, label_list = Load_image_label_txt(image_txt, 2, shuffle)  # 获取一个label
    with tf.io.TFRecordWriter(image_tfrecord_output) as writer:
        for i, [image_name, labels] in enumerate(zip(images_list, label_list)):
            image = Read_mul_view(image_name)
            image_raw = image.tobytes()
            if i % log == 0 or i == len(images_list) - 1:
                print('--------processing:{},{}--------'.format(i, image_name))
            label = int(labels)
            feeature_dict = {
                'mul_view_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'mul_view_b': tf.train.Feature(int64_list=tf.train.Int64List(value=[12])),
                'mul_view_h': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
                'mul_view_w': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
                'mul_view_d': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
            example = tf.train.Example(features=tf.train.Features(feature=feeature_dict))
            writer.write(example.SerializeToString())
    return


if __name__ == '__main__':
    shuffle = True
    log = 10  # 打印间隔
    # 生产train——TFrecords
    name = "人工分型_total_list"
    train_txt = f'{name}.txt'
    train_tfrecord_output = f'data/{name}_500.tfrecords'
    # Create_tfrecord(name, train_txt, train_tfrecord_output, shuffle, log)
    tf_nums = Get_tfrecords_nums(train_tfrecord_output)
    print(f"successed create {train_tfrecord_output},nums={tf_nums}")
    # 生产val——TFrecords
    # name = "model200"
    # val_txt = f'{name}_list.txt'
    # val_tfrecord_output = f'data/{name}.tfrecords'
    # Create_tfrecord(name, val_txt, val_tfrecord_output, shuffle, log)
    # tf_nums = Get_tfrecords_nums(val_tfrecord_output)
    # print(f"successed create {val_tfrecord_output},nums={tf_nums}")

    # dataset = Read_pcd_tfrecords(train_tfrecord_output)
    # for i, (pcl_pcl, pcl_label) in enumerate(dataset):
    #     print(i, "   ", pcl_label)
    #     pcl_pcl = pcl_pcl.numpy()
    #     pcl_pcl = pcl_pcl.astype(np.float32)
    #     print(pcl_pcl)
    #     print(pcl_pcl.shape)
