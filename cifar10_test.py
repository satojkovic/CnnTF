#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import cifar10
import os
import tensorflow as tf


class TestCifar10(tf.test.TestCase):
    def test_maybe_download(self):
        cifar10.maybe_download_and_extract()
        self.assertEqual(
            True, os.path.exists('/tmp/cifar10_data/cifar-10-binary.tar.gz'))

    def test_distorted_inputs(self):
        images, labels = cifar10.distorted_inputs()
        with self.test_session() as sess:
            batch_size, image_size, _, _ = images.shape
            self.assertEqual(128, batch_size)
            self.assertEqual(24, image_size)


if __name__ == '__main__':
    unittest.main()
