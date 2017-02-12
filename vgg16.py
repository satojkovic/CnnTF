#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

# parameters
patch_size = 3
image_size = 224
pooled_image_size = image_size // 2**5  # max pooling x 5
n_img_channels = 3

channels = {
    'input': 3,
    'conv1': 64,
    'conv2': 128,
    'conv3': 256,
    'conv4': 512,
    'conv5': 512,
    'fc1': 4096,
    'fc2': 4096,
    'fc3': 1000,
}


def vgg16(x, weights, biases, initial_weights):
    # paramters
    params = []

    # conv layer1_1 with relu
    with tf.name_scope('conv1_1') as scope:
        conv1_1 = tf.nn.conv2d(
            x, weights['conv1_1'], [1, 1, 1, 1], padding='SAME')
        conv1_1 = tf.nn.bias_add(conv1_1, biases['conv1_1'])
        conv1_1 = tf.nn.relu(conv1_1, name=scope)
    # add paramters of conv1_1 layer
    params += [weights['conv1_1'], biases['conv1_1']]

    # conv layer1_2 with relu
    with tf.name_scope('conv1_2') as scope:
        conv1_2 = tf.nn.conv2d(
            conv1_1, weights['conv1_2'], [1, 1, 1, 1], padding='SAME')
        conv1_2 = tf.nn.bias_add(conv1_2, biases['conv1_2'])
        conv1_2 = tf.nn.relu(conv1_2, name=scope)
    # add paramters of conv1_2 layer
    params += [weights['conv1_2'], biases['conv1_2']]

    # max pooling
    pool1 = tf.nn.max_pool(
        conv1_2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1')

    # conv laer2_1 with relu
    with tf.name_scope('conv2_1') as scope:
        conv2_1 = tf.nn.conv2d(
            pool1, weights['conv2_1'], [1, 1, 1, 1], padding='SAME')
        conv2_1 = tf.nn.bias_add(conv2_1, biases['conv2_1'])
        conv2_1 = tf.nn.relu(conv2_1)
    # add paramters of conv2_1 layer
    params += [weights['conv2_1'], biases['conv2_1']]

    # conv layer2_2 with relu
    with tf.name_scope('conv2_2') as scope:
        conv2_2 = tf.nn.conv2d(
            conv2_1, weights['conv2_2'], [1, 1, 1, 1], padding='SAME')
        conv2_2 = tf.nn.bias_add(conv2_2, biases['conv2_2'])
        conv2_2 = tf.nn.relu(conv2_2)
    # add paramters of conv2_2 layer
    params += [weights['conv2_2'], biases['conv2_2']]

    # max pooling
    pool2 = tf.nn.max_pool(
        conv2_2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2')

    # conv layer3_1 with relu
    with tf.name_scope('conv3_1') as scope:
        conv3_1 = tf.nn.conv2d(
            pool2, weights['conv3_1'], [1, 1, 1, 1], padding='SAME')
        conv3_1 = tf.nn.bias_add(conv3_1, biases['conv3_1'])
        conv3_1 = tf.nn.relu(conv3_1)
    # add paramters of conv3_1 layer
    params += [weights['conv3_1'], biases['conv3_1']]

    # conv layer3_2 with relu
    with tf.name_scope('conv3_2') as scope:
        conv3_2 = tf.nn.conv2d(
            conv3_1, weights['conv3_2'], [1, 1, 1, 1], padding='SAME')
        conv3_2 = tf.nn.bias_add(conv3_1, biases['conv3_2'])
        conv3_2 = tf.nn.relu(conv3_2)
    # add paramters of conv3_2 layer
    params += [weights['conv3_2'], biases['conv3_2']]

    # conv layer3_3 with relu
    with tf.name_scope('conv3_3') as scope:
        conv3_3 = tf.nn.conv2d(
            conv3_2, weights['conv3_3'], [1, 1, 1, 1], padding='SAME')
        conv3_3 = tf.nn.bias_add(conv3_3, biases['conv3_3'])
        conv3_3 = tf.nn.relu(conv3_3)
    # add paramters of conv3_3 layer
    params += [weights['conv3_3'], biases['conv3_3']]

    # max pooling
    pool3 = tf.nn.max_pool(
        conv3_3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool3')
    # conv layer4_1 with relu
    with tf.name_scope('conv4_1') as scope:
        conv4_1 = tf.nn.conv2d(
            pool3, weights['conv4_1'], [1, 1, 1, 1], padding='SAME')
        conv4_1 = tf.nn.bias_add(conv4_1, biases['conv4_1'])
        conv4_1 = tf.nn.relu(conv4_1)
    # add paramters of conv4_1 layer
    params += [weights['conv4_1'], biases['conv4_1']]

    # conv layer4_2 with relu
    with tf.name_scope('conv4_2') as scope:
        conv4_2 = tf.nn.conv2d(
            conv4_1, weights['conv4_2'], [1, 1, 1, 1], padding='SAME')
        conv4_2 = tf.nn.bias_add(conv4_2, biases['conv4_2'])
        conv4_2 = tf.nn.relu(conv4_2)
    # add paramters of conv4_2 layer
    params += [weights['conv4_2'], biases['conv4_2']]

    # conv layer4_3 with relu
    with tf.name_scope('conv4_3') as scope:
        conv4_3 = tf.nn.conv2d(
            conv4_2, weights['conv4_3'], [1, 1, 1, 1], padding='SAME')
        conv4_3 = tf.nn.bias_add(conv4_3, biases['conv4_3'])
        conv4_3 = tf.nn.relu(conv4_3)
    # add paramters of conv4_3 layer
    params += [weights['conv4_3'], biases['conv4_3']]

    # max pooling
    pool4 = tf.nn.max_pool(
        conv4_3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool4')

    # conv layer5_1 with relu
    with tf.name_scope('conv5_1') as scope:
        conv5_1 = tf.nn.conv2d(
            pool4, weights['conv5_1'], [1, 1, 1, 1], padding='SAME')
        conv5_1 = tf.nn.bias_add(conv5_1, biases['conv5_1'])
        conv5_1 = tf.nn.relu(conv5_1)
    # add paramters of conv5_1 layer
    params += [weights['conv5_1'], biases['conv5_1']]

    # conv layer5_2 with relu
    with tf.name_scope('conv5_2') as scope:
        conv5_2 = tf.nn.conv2d(
            conv5_1, weights['conv5_2'], [1, 1, 1, 1], padding='SAME')
        conv5_2 = tf.nn.bias_add(conv5_2, biases['conv5_2'])
        conv5_2 = tf.nn.relu(conv5_2)
    # add paramters of conv5_2 layer
    params += [weights['conv5_2'], biases['conv5_2']]

    # conv layer5_3 with relu
    with tf.name_scope('conv5_3') as scope:
        conv5_3 = tf.nn.conv2d(
            conv5_2, weights['conv5_3'], [1, 1, 1, 1], padding='SAME')
        conv5_3 = tf.nn.bias_add(conv5_3, biases['conv5_3'])
        conv5_3 = tf.nn.relu(conv5_3)
    # add paramters of conv5_3 layer
    params += [weights['conv5_3'], biases['conv5_3']]

    # max pooling
    pool5 = tf.nn.max_pool(
        conv5_3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool5')

    # fc layer1
    with tf.name_scope('fc1') as scope:
        pool5_flat = tf.reshape(pool5, [
            -1, pooled_image_size * pooled_image_size * channels['conv5']
        ])
        fc1 = tf.nn.bias_add(
            tf.matmul(pool5_flat, weights['fc1']), biases['fc1'])
        fc1 = tf.nn.relu(fc1)
    # add paramters of fc1 layer
    params += [weights['fc1'], biases['fc1']]

    # fc layer2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.nn.bias_add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
        fc2 = tf.nn.relu(fc2)
    # add paramters of fc2 layer
    params += [weights['fc2'], biases['fc2']]

    # fc layer3
    with tf.name_scope('fc3') as scope:
        fc3 = tf.nn.bias_add(tf.matmul(fc2, weights['fc3']), biases['fc3'])
    # add paramters of fc3 layer
    params += [weights['fc3'], biases['fc3']]

    # weights initialize
    if initial_weights is not None:
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

        # softmax layer
    pred = tf.nn.softmax(fc3)

    return pred, params


def main():
    # load weights
    if len(sys.argv) > 1:
        f = np.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files)]
    else:
        initial_weights = None

    # weights and biases
    weights = {
        'conv1_1':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['input'], channels['conv1']],
                stddev=0.1),
            name='weights'),
        'conv1_2':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv1'], channels['conv1']],
                stddev=0.1),
            name='weights'),
        'conv2_1':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv1'], channels['conv2']],
                stddev=0.1),
            name='weights'),
        'conv2_2':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv2'], channels['conv2']],
                stddev=0.1),
            name='weights'),
        'conv3_1':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv2'], channels['conv3']],
                stddev=0.1),
            name='weights'),
        'conv3_2':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv3'], channels['conv3']],
                stddev=0.1),
            name='weights'),
        'conv3_3':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv3'], channels['conv3']],
                stddev=0.1),
            name='weights'),
        'conv4_1':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv3'], channels['conv4']],
                stddev=0.1),
            name='conv4_1'),
        'conv4_2':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv4'], channels['conv4']],
                stddev=0.1),
            name='weights'),
        'conv4_3':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv4'], channels['conv4']],
                stddev=0.1),
            name='weights'),
        'conv5_1':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv4'], channels['conv5']],
                stddev=0.1),
            name='weights'),
        'conv5_2':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv5'], channels['conv5']],
                stddev=0.1),
            name='weights'),
        'conv5_3':
        tf.Variable(
            tf.truncated_normal(
                [patch_size, patch_size, channels['conv5'], channels['conv5']],
                stddev=0.1),
            name='weights'),
        'fc1':
        tf.Variable(
            tf.truncated_normal(
                [
                    pooled_image_size * pooled_image_size * channels['conv5'],
                    channels['fc1']
                ],
                stddev=0.1),
            name='weights'),
        'fc2':
        tf.Variable(
            tf.truncated_normal(
                [channels['fc1'], channels['fc2']], stddev=0.1),
            name='weights'),
        'fc3':
        tf.Variable(
            tf.truncated_normal(
                [channels['fc2'], channels['fc3']], stddev=0.1),
            name='weights'),
    }
    biases = {
        'conv1_1': tf.Variable(tf.constant(0.0, shape=[channels['conv1']])),
        'conv1_2': tf.Variable(tf.constant(0.0, shape=[channels['conv1']])),
        'conv2_1': tf.Variable(tf.constant(0.0, shape=[channels['conv2']])),
        'conv2_2': tf.Variable(tf.constant(0.0, shape=[channels['conv2']])),
        'conv3_1': tf.Variable(tf.constant(0.0, shape=[channels['conv3']])),
        'conv3_2': tf.Variable(tf.constant(0.0, shape=[channels['conv3']])),
        'conv3_3': tf.Variable(tf.constant(0.0, shape=[channels['conv3']])),
        'conv4_1': tf.Variable(tf.constant(0.0, shape=[channels['conv4']])),
        'conv4_2': tf.Variable(tf.constant(0.0, shape=[channels['conv4']])),
        'conv4_3': tf.Variable(tf.constant(0.0, shape=[channels['conv4']])),
        'conv5_1': tf.Variable(tf.constant(0.0, shape=[channels['conv5']])),
        'conv5_2': tf.Variable(tf.constant(0.0, shape=[channels['conv5']])),
        'conv5_3': tf.Variable(tf.constant(0.0, shape=[channels['conv5']])),
        'fc1': tf.Variable(tf.constant(1.0, shape=[channels['fc1']])),
        'fc2': tf.Variable(tf.constant(1.0, shape=[channels['fc2']])),
        'fc3': tf.Variable(tf.constant(1.0, shape=[channels['fc3']])),
    }

    # construct a model
    x = tf.placeholder(
        tf.float32, shape=[None, image_size, image_size, n_img_channels])
    vgg, params = vgg16(x, weights, biases, initial_weights)


if __name__ == '__main__':
    main()
