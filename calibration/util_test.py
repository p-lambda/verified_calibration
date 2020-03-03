import unittest
from parameterized import parameterized
from utils import *
from utils import _get_ce

import collections


def multiset_equal(list1, list2):
    return collections.Counter(list1) == collections.Counter(list2)


def list_to_tuple(l):
    if not isinstance(l, list):
        return l
    return tuple(list_to_tuple(x) for x in l)


class TestUtilMethods(unittest.TestCase):

    def test_split(self):
        self.assertEqual(split([1, 3, 2, 4], parts=2), [[1, 3], [2, 4]])
        self.assertEqual(split([1], parts=1), [[1]])
        self.assertEqual(split([2, 3], parts=1), [[2, 3]])
        self.assertEqual(split([2, 3], parts=2), [[2], [3]])
        self.assertEqual(split([1, 2, 3], parts=1), [[1, 2, 3]])
        self.assertEqual(split([1, 2, 3], parts=2), [[1, 2], [3]])
        self.assertEqual(split([1, 2, 3], parts=3), [[1], [2], [3]])

    def test_get_1_equal_bin(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_bins(probs, num_bins=1)
        self.assertEqual(bins, [1.0])

    def test_get_2_equal_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_bins(probs, num_bins=2)
        self.assertEqual(bins, [0.4, 1.0])

    def test_get_3_equal_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_bins(probs, num_bins=3)
        self.assertEqual(bins, [0.3, 0.5, 1.0])

    def test_get_1_equal_prob_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_prob_bins(probs, num_bins=1)
        self.assertEqual(bins, [1.0])

    def test_get_2_equal_prob_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_equal_prob_bins(probs, num_bins=2)
        self.assertEqual(bins, [0.5, 1.0])

    def test_get_discrete_bins(self):
        probs = [0.3, 0.5, 0.2, 0.3, 0.5, 0.7]
        bins = get_discrete_bins(probs)
        self.assertEqual(bins, [0.25, 0.4, 0.6, 1.0])

    def test_enough_duplicates(self):
        probs = np.array([0.1, 0.3, 0.5])
        self.assertFalse(enough_duplicates(probs))
        probs = np.array([0.1, 0.1, 0.5])
        self.assertFalse(enough_duplicates(probs))
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.assertTrue(enough_duplicates(probs))

    def test_get_bin(self):
        bins = [0.2, 0.5, 1.0]
        self.assertEqual(get_bin(0.0, bins), 0)
        self.assertEqual(get_bin(0.19, bins), 0)
        self.assertEqual(get_bin(0.21, bins), 1)
        self.assertEqual(get_bin(0.49, bins), 1)
        self.assertEqual(get_bin(0.51, bins), 2)
        self.assertEqual(get_bin(1.0, bins), 2)

    def test_get_bin_size_1(self):
        bins = [1.0]
        self.assertEqual(get_bin(0.0, bins), 0)
        self.assertEqual(get_bin(0.5, bins), 0)
        self.assertEqual(get_bin(1.0, bins), 0)

    def test_bin(self):
        data = [(0.3, 1.0), (0.5, 0.0), (0.2, 1.0), (0.3, 0.0), (0.5, 1.0), (0.7, 0.0)]
        bins = [0.4, 1.0]
        binned_data = tuple(np.array(bin(data, bins)).tolist())
        self.assertTrue(multiset_equal(
            list_to_tuple(binned_data[0]), ((0.3, 1.0), (0.2, 1.0), (0.3, 0.0))))
        self.assertTrue(multiset_equal(
            list_to_tuple(binned_data[1]), ((0.5, 1.0), (0.5, 0.0), (0.7, 0.0))))

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

    @parameterized.expand([
        [1, 4/9*0.25+5/9*0.2],
        [2, (4/9*(0.25**2)+5/9*(0.2**2))**(1/2.0)],
        [3, (4/9*(0.25**3)+5/9*(0.2**3))**(1/3.0)],
    ])
    def test_get_ce(self, p, true_ce):
        logits = [0.5, 0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.6, 0.6]
        labels = [0, 1, 0, 1, 0, 1, 1, 1, 0]
        pred_ce = _get_ce(logits, labels, p, debias=False, num_bins=None,
            binning_scheme=get_discrete_bins)
        self.assertAlmostEqual(pred_ce, true_ce)
        pred_ce = _get_ce(logits, labels, p, debias=True, num_bins=None,
            binning_scheme=get_discrete_bins)
        self.assertLess(pred_ce, true_ce)



if __name__ == '__main__':
    unittest.main()
