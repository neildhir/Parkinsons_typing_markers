import unittest

import pandas as pd
from haberrspd.preprocess import make_character_compression_time_sentence, backspace_corrector


class TestPreprocessing(unittest.TestCase):

    def setUp(self):

        # Test cases
        self.base_sequence = list('pesto')
        self.raw_character_sequence = pd.Series(self.base_sequence)
        self.raw_timestamps = pd.Series([2, 4, 5, 8, 10])
        # Sequences with leading and trailing backspaces
        self.backspace_error_one = ['backspace'] + self.base_sequence
        self.backspace_error_two = self.base_sequence + ['backspace']
        self.backspace_error_three = 2 * ['backspace'] + self.base_sequence
        self.backspace_error_four = self.base_sequence + 2 * ['backspace']

    def test_long_format_construction(self):
        """
        Long-format character construction should be done correctly.
        This test ensure that the construction is commensurate with the intention.
        """
        target = list('ppessstt')  # Expected output
        output = make_character_compression_time_sentence(
            self.raw_timestamps,
            self.raw_character_sequence,
            time_redux_fact=1)
        # Test assertion
        self.assertEqual(output, target)

    def test_leading_backspace_removal(self):
        """
        This method tests if we can accurately remove leading backspaces.
        """

        # Single leading backspace
        output, indices = backspace_corrector(self.backspace_error_one)
        self.assertEqual(output, self.base_sequence)
        # Single leading backspace
        output, indices = backspace_corrector(self.backspace_error_three)
        self.assertEqual(output, self.base_sequence)

    def test_leading_trailing_removal(self):
        """
        This method tests if we can accurately remove trailing backspaces.
        """

        # Single leading backspace
        output, indices = backspace_corrector(self.backspace_error_two)
        self.assertEqual(output, self.base_sequence)
        # Single leading backspace
        output, indices = backspace_corrector(self.backspace_error_four)
        self.assertEqual(output, self.base_sequence[:-1])


if __name__ == '__main__':
    unittest.main()
