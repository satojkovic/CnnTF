#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cifar
import os
import tensorflow as tf


class TestCifar10(tf.test.TestCase):
    def test_maybe_download(self):
        cifar.maybe_download_and_extract()
        self.assertEqual(
            True, os.path.exists('/tmp/cifar_data/cifar-10-binary.tar.gz'))


if __name__ == '__main__':
    tf.test.main()
