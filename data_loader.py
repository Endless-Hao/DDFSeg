"""Code for training DDFSeg."""
#Peichenhao
import tensorflow as tf
import numpy as np

BATCH_SIZE = 4


def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        'dsize_dim0': tf.FixedLenFeature([], tf.int64),
        'dsize_dim1': tf.FixedLenFeature([], tf.int64),
        'dsize_dim2': tf.FixedLenFeature([], tf.int64),
        'lsize_dim0': tf.FixedLenFeature([], tf.int64),
        'lsize_dim1': tf.FixedLenFeature([], tf.int64),
        'lsize_dim2': tf.FixedLenFeature([], tf.int64),
        'data_vol': tf.FixedLenFeature([], tf.string),
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 1]
    volume_size = [256, 256, 1]
    label_size = [256, 256, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 0], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 4)

    return tf.expand_dims(data_vol[:, :, 0], axis=2), batch_y


def _load_samples(source_pth, target_pth):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    data_vola, label_vola = _decode_samples(imagea_list, shuffle=True)
    data_volb, label_volb = _decode_samples(imageb_list, shuffle=True)

    return data_vola, data_volb, label_vola, label_volb


def load_data(source_pth, target_pth, do_shuffle=True):

    image_i, image_j, gt_i, gt_j = _load_samples(source_pth, target_pth)
    print(image_i.shape)
    image_i = tf.concat((image_i,image_i,image_i), axis=2)
    image_j = tf.concat((image_j,image_j,image_j), axis=2)
    print(image_i.shape)
    print(image_j.shape)
    print(gt_i.shape)
    print(gt_j.shape)

    # Batch
    if do_shuffle is True:
        images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch([image_i, image_j, gt_i, gt_j], BATCH_SIZE, 500, 100)
    else:
        images_i, images_j, gt_i, gt_j = tf.train.batch([image_i, image_j, gt_i, gt_j], batch_size=BATCH_SIZE, num_threads=1, capacity=500)

    return images_i, images_j, gt_i, gt_j

def _decode_test_samples(image_list, shuffle=False):
    decomp_feature = {
        'dsize_dim0': tf.FixedLenFeature([], tf.int64),
        'dsize_dim1': tf.FixedLenFeature([], tf.int64),
        'dsize_dim2': tf.FixedLenFeature([], tf.int64),
        'lsize_dim0': tf.FixedLenFeature([], tf.int64),
        'lsize_dim1': tf.FixedLenFeature([], tf.int64),
        'lsize_dim2': tf.FixedLenFeature([], tf.int64),
        'data_vol': tf.FixedLenFeature([], tf.string),
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [128, 128, 1]
    volume_size = [128, 128, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=False)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    return tf.expand_dims(data_vol[:, :, 0], axis=2)


def _load_test_samples(source_pth):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    data_vola= _decode_test_samples(imagea_list, shuffle=False)

    return data_vola


def load_testdata(source_pth, do_shuffle=False):

    image_i= _load_test_samples(source_pth)
    print(image_i.shape)

    image_i = tf.concat((image_i,image_i,image_i), axis=2)
    print(image_i.shape)

    # Batch
    if do_shuffle is True:
        images_i= tf.train.shuffle_batch([image_i], BATCH_SIZE, 500, 100)
    else:
        images_i = tf.train.batch([image_i], batch_size=BATCH_SIZE, num_threads=1, capacity=500)

    return images_i
