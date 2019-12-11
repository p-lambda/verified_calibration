import unittest
from parameterized import parameterized
from utils import *

import collections

def multiset_equal(list1, list2):
    return collections.Counter(list1) == collections.Counter(list2)

class TestUtilMethods(unittest.TestCase):

    def test_split(self):
        self.assertEqual(split([1, 3, 2, 4], parts=2), [[1, 3], [2, 4]])

    def test_get_1_equal_bin(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_bins(probs, num_bins=1)
        self.assertEqual(bins, [1.0])

    def test_get_2_equal_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_bins(probs, num_bins=2)
        self.assertEqual(bins, [0.4, 1.0])

    def test_get_bin(self):
        bins = [0.2, 0.5, 1.0]
        self.assertEqual(get_bin(0.0, bins), 0)
        self.assertEqual(get_bin(0.19, bins), 0)
        self.assertEqual(get_bin(0.21, bins), 1)
        self.assertEqual(get_bin(0.49, bins), 1)
        self.assertEqual(get_bin(0.51, bins), 2)
        self.assertEqual(get_bin(1.0, bins), 2)

    def test_bin(self):
        data = [(0.3, 1.0), (0.5, 0.0), (0.2, 1.0), (0.3, 0.0), (0.5, 1.0), (0.7, 0.0)]
        bins = [0.4, 1.0]
        binned_data = tuple(np.array(bin(data, bins)).tolist())
        print(binned_data)
        self.assertTrue(multiset_equal(list(binned_data[0]), ((0.3, 1.0), (0.2, 1.0), (0.3, 0.0))))
        self.assertTrue(multiset_equal(list(binned_data[1]), ((0.5, 1.0), (0.5, 0.0), (0.7, 0.0))))

    @parameterized.expand([
        [[(0.3, 0.5)], -0.2],
        [[(0.5, 0.3)], 0.2],
        [[(0.3, 0.5), (0.8, 0.4)], 0.1],
        [[(0.3, 0.5), (0.8, 0.4), (0.4, 0.0)], 0.2]
    ])
    def test_difference_mean(self, data, true_value):
        self.assertAlmostEqual(difference_mean(data), true_value)


    @parameterized.expand([
        [[[(0.3, 0.5)]], [1.0]],
        [[[(0.3, 0.5)], [(0.4, 0.7)]], [0.5, 0.5]],
        [[[(0.3, 0.5)], [(0.4, 0.7)], [(0.0, 1.0), (0.6, 0.0)]], [0.25, 0.25, 0.5]],
    ])
    def test_get_bin_probs(self, binned_data, probs):
        self.assertAlmostEqual(get_bin_probs(binned_data), probs)

    @parameterized.expand([
        [[[(0.3, 1.0)]], 1, 0.7],
        [[[(0.3, 0.0)]], 1, 0.3],
        [[[(0.3, 1.0)]], 2, 0.7],
        [[[(0.3, 1.0)], [(0.6, 0.0)]], 1, 0.65],
        [[[(0.3, 1.0)], [(0.6, 0.0), (0.6, 1.0)]], 1, 0.3],
        [[[(0.3, 1.0)], [(0.6, 0.0)]], 2, 0.6519202405],
    ])
    def test_plugin_ce(self, binned_data, power, true_ce):
        self.assertAlmostEqual(plugin_ce(binned_data, power), true_ce)


if __name__ == '__main__':
    unittest.main()