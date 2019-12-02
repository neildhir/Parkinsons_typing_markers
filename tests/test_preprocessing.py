import itertools
import unittest

import numpy as np
import pandas as pd

from haberrspd.preprocess import (universal_backspace_implementer,
                                  make_long_format_sentence,
                                  sentence_level_pause_correction)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):

        # Test cases
        self.base_sequence = list('pesto')
        self.raw_character_sequence = pd.Series(self.base_sequence)
        self.compression_times = pd.Series([1., 2., 2., 4., 3.])

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

        # Dataframe objects for tests
        test_keys = ['liverpool', 'barcelona']  # change any word but don't add words
        self.df = pd.DataFrame(
            {
                'timestamp': np.hstack([
                    np.linspace(1, 15, len(test_keys[0]), dtype=int),
                    np.linspace(1, 25, len(test_keys[1]), dtype=int),
                    np.linspace(1, 17, len(test_keys[0]), dtype=int),
                    np.linspace(1, 33, len(test_keys[0]), dtype=int),
                ]),
                'key': list(''.join(map(str, test_keys*2))),
                'participant_id': ['1a'] * len(test_keys[0]+test_keys[1]) + ['2a'] * len(test_keys[0]+test_keys[1]),
                'sentence_id': list(itertools.chain(*[[x]*y for x, y in zip([0, 1], [len(test_keys[0]), len(test_keys[1])])]*2))
            }
        )
        # Add a special timestamp to test if the response-time outlier replacement works
        self.df.iloc[-1, 0] = 1000

    def test_long_format_construction(self):
        """
        Long-format character construction should be done correctly.
        This test ensure that the construction is commensurate with the intention.
        """
        target = list('ppeessssttt')  # Expected output
        output = make_long_format_sentence(
            self.compression_times,
            self.raw_character_sequence,
            time_redux_fact=1)
        # Test assertion
        self.assertEqual(output, target)

    def test_leading_backspace_removal(self):
        """
        This method tests if we can accurately remove leading backspaces.
        """

        # Single leading backspace
        output, _ = universal_backspace_implementer(self.backspace_error_one)
        self.assertEqual(output, self.base_sequence)
        # Multiple leading backspace
        output, _ = universal_backspace_implementer(self.backspace_error_three)
        self.assertEqual(output, self.base_sequence)

    def test_leading_trailing_removal(self):
        """
        This method tests if we can accurately remove trailing backspaces.
        """

        # Single leading backspace
        output, _ = universal_backspace_implementer(self.backspace_error_two)
        self.assertEqual(output, self.base_sequence)
        # Multiple trailing backspaces
        output, _ = universal_backspace_implementer(self.backspace_error_four)
        # Output == ['p','e','s','t']
        self.assertEqual(output, self.base_sequence[:-1])

    def test_compound_error_sequence(self):
        """
        Test a complex sequence of characters and backspaces.
        """
        output, indices = universal_backspace_implementer(self.complex_error_one)
        self.assertEqual(output, list('to tastes very ni'))
        # Check that deletion indices are correct
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5, 10, 24, 25, 26, 27, 28])

    def test_backspace_replacement(self):
        output, _ = universal_backspace_implementer(self.backspace_error_one, invokation_type=-1)
        self.backspace_error_one[0] = '£'
        # Should return a sequence in which the backspace has been replaced with a £ char.
        self.assertEqual(output, self.backspace_error_one)

    def test_response_time_correction(self):
        """
        This function tests the response-time correction functionality. Subjects pause
        when they type (sometimes), this causes irregularities in the natural response-time
        distribution of natural typing. This function tests if our filter/replacement
        procedure, produces the correct values.
        """

        # Invoke correction method, keeping all defaults but one
        out = sentence_level_pause_correction(self.df, char_count_response_threshold=3)

        self.assertEqual(out[0]['1a'].all(), pd.Series([np.nan, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0]).all())
        self.assertEqual(out[1]['1a'].all(), pd.Series([np.nan, 3., 3., 3., 3., 3., 3., 3., 3.]).all())
        self.assertEqual(out[0]['2a'].all(), pd.Series([np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).all())
        # Check that the outlier value has been sufficiently replaced
        self.assertGreater(self.df.iloc[-1, 0], out[1]['2a'].iloc[-1])


if __name__ == '__main__':
    unittest.main()
