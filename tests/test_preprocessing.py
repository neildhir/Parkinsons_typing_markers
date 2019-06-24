import unittest

from pandas import Series

from haberrspd.preprocess import make_character_compression_time_sentence


class TestPreprocessing(unittest.TestCase):

    def test_long_format_construction(self):
        """
        Long-format character construction should be done correctly.
        This test ensure that the construction is commensurate with the intention.
        """
        char_sequence = Series(list('pesto'))
        timestamps = Series([2, 4, 5, 8, 10])
        target = list('ppessstt')
        output = make_character_compression_time_sentence(timestamps, char_sequence, time_redux_fact=1)
        # Test assertion
        self.assertEqual(output, target)
