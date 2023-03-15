import numpy as np
import pytest

from co_monitor.emissivity.reader import Emissivity


list_of_nrs = [1, 22, 50, 16, 7, 19, 100, 4, 6, 211]


def test_find_nearest():
    index = Emissivity._find_nearest(list_of_nrs, 75)
    expected = 2
    assert index == expected

    index = Emissivity._find_nearest(list_of_nrs, 76)
    expected = 6
    assert index == expected

    index = Emissivity._find_nearest(list_of_nrs, 0)
    expected = 0
    assert index == expected
