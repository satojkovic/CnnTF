#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cifar10
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs""",
                           """and checkpoint.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)

        train_op = cifar10.train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=
                                                FLAGS.log_device_placement))
        sess.run(init)


def main():
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    main()
