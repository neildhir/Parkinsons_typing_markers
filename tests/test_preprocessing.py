import unittest

import pandas as pd

from haberrspd.preprocess import (backspace_corrector,
                                  make_character_compression_time_sentence,
                                  sentence_level_pause_correction)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):

        # Test cases
        self.base_sequence = list('pesto')
        self.raw_character_sequence = pd.Series(self.base_sequence)
        self.raw_timestamps = pd.Series([2, 4, 5, 8, 10])

        error = 'backspace'

        # Sequences with leading and trailing backspaces
        self.backspace_error_one = [error] + self.base_sequence
        self.backspace_error_two = self.base_sequence + [error]
        self.backspace_error_three = 2 * [error] + self.base_sequence
        self.backspace_error_four = self.base_sequence + 2 * [error]

        # More complex error sequences
        # Clean: 'pesto tastes very nice'
        self.complex_error_one = ['p',
                                  'e',
                                  'backspace',
                                  'backspace',
                                  'backspace',
                                  's',
                                  't',
                                  'o',
                                  ' ',
                                  't',
                                  'backspace',
                                  'a',
                                  's',
                                  't',
                                  'e',
                                  's',
                                  ' ',
                                  'v',
                                  'e',
                                  'r',
                                  'y',
                                  ' ',
                                  'n',
                                  'i',
                                  'c',
                                  'e',
                                  'backspace',
                                  'backspace',
                                  'backspace']

        self.df_small = pd.DataFrame(
            {'timestamp': [1, 4, 7, 2, 5, 8, 2, 4, 9, 3, 5, 8],
             'key': list('car') + list('toy') + list('car') + list('toy'),
             'participant_id': ['1a'] * 6 + ['2a'] * 6,
             'sentence_id': [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3
             }
        )

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
        output, _ = backspace_corrector(self.backspace_error_one)
        self.assertEqual(output, self.base_sequence)
        # Multiple leading backspace
        output, _ = backspace_corrector(self.backspace_error_three)
        self.assertEqual(output, self.base_sequence)

    def test_leading_trailing_removal(self):
        """
        This method tests if we can accurately remove trailing backspaces.
        """

        # Single leading backspace
        output, _ = backspace_corrector(self.backspace_error_two)
        self.assertEqual(output, self.base_sequence)
        # Multiple trailing backspaces
        output, _ = backspace_corrector(self.backspace_error_four)
        # Output == ['p','e','s','t']
        self.assertEqual(output, self.base_sequence[:-1])

    def test_compound_error_sequence(self):
        """
        Test a complex sequence of characters and backspaces.
        """
        output, indices = backspace_corrector(self.complex_error_one)
        self.assertEqual(output, list('to tastes very ni'))
        # Check that deletion indices are correct
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5, 10, 24, 25, 26, 27, 28])

    def test_backspace_replacement(self):
        output, _ = backspace_corrector(self.backspace_error_one, invokation_type=-1)
        self.backspace_error_one[0] = '£'
        # Should return a sequence in which the backspace has been replaced with a £ char.
        self.assertEqual(output, self.backspace_error_one)

    def test_response_time_correction(self):
        pass


if __name__ == '__main__':
    unittest.main()
