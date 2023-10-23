import unittest

from ..utils import split_list


class util_test(unittest.TestCase):
    def setUp(self):
        pass

    def test_list_split(self):
        temp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]

        self.assertEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [14]], split_list(temp, 3, strip_remains=False))
        self.assertEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split_list(temp, 3, strip_remains=True))
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7, 8, 9, 14]], split_list(temp, 20, strip_remains=False))
        self.assertEqual([], split_list(temp, 20, strip_remains=True))
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7, 8, 9, 14]], split_list(temp, 10, strip_remains=False))


if __name__ == "__main__":
    unittest.main()
