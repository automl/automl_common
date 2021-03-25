# -*- encoding: utf-8 -*-
import os
import unittest

from common.utils.common import check_pid


def test_check_pid():
    our_pid = os.getpid()

    exists = check_pid(our_pid)
    assert exists
    our_pid = -11000  # We hope this pid does not exist
    exists = check_pid(our_pid)
    assert not exists
