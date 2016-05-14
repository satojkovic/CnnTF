#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import cifar10
import os


class TestCifar10(unittest.TestCase):
    def test_maybe_download(self):
        cifar10.maybe_download_and_extract()
        self.assertEqual(
            True, os.path.exists('/tmp/cifar10_data/cifar-10-binary.tar.gz'))


if __name__ == '__main__':
    unittest.main()
