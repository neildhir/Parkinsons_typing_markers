"""
From Elisa's code base: https://github.com/elisaF/typing-classification/blob/master/keyboard_distance.py
"""


# coding=utf-8
from __future__ import division
import numpy as np
import src.helper_functions as helper
import logging

logger = logging.getLogger("keyboard_distance")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("keyboard_distance.log")
logger.addHandler(fh)

# QWERTY Keyboard Map for US English
#
#   | 0   1   2   3   4   5   6   7   8   9   10  11  12  13
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# 0 | ` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0 | - | = |   |
# 1 |   | q | w | e | r | t | y | u | i | o | p | [ | ] | \ |
# 2 |   | a | s | d | f | g | h | j | k | l | ; | ' |   |   |
# 3 |   | z | x | c | v | b | n | m | , | . | / |   |   |   |
# 4 |   |   |   | _ | _ | _ | _ | _ |   |   |   |   |   |   |
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
#
# QWERTY Keyboard Map for Spain
#
#   | 0   1   2   3   4   5   6   7   8   9   10  11  12
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+
# 0 | º | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0 | ' | ¡ |
#   | ª | ! | " | · | $ | % | & | / | ( | ) | = | ? | ¿ |
# 1 |   | q | w | e | r | t | y | u | i | o | p | ` | + |
#   |   | Q | W | E | R | T | Y | U | I | O | P | ^ | * |
# 2 |   | a | s | d | f | g | h | j | k | l | ñ | ´ | ç |
#   |   | A | S | D | F | G | H | J | K | L | Ñ | ¨ | Ç |
# 3 | < | z | x | c | v | b | n | m | , | . | - |   |   |
#   | > | Z | X | C | V | B | N | M | ; | : | _ |   |   |
# 4 |   |   |   | _ | _ | _ | _ | _ |   |   |   |   |   |  <== space bar!
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+
#
# Distance between two keys:
#  -if keys are orthogonal to each other: 1
#  -if keys are on secondary diagonal (top-right to bottom-left): 1
#  -other: sqrt(x^2 + y^2), where x is horizontal distance and y is vertical distance
#  -if keys are not on map, then return the distance to the average location (middle of the keyboard)
#
# NOTE: Characters with diacritics that are not on the keyboard are mapped to their
#       diacriticless counterparts (e.g. é -> e)
#
# Characters  that are not on the map:
#  -ø, æ, ß
#  -Chinese, Greek, Cyrillic characters


class KeyboardDistance:
    def __init__(self, language="english"):
        self.language = language
        self.qwerty_grid = None
        self.QWERTY_grid = None
        self.average_x, self.average_y = None, None
        self.initialize(language)

    def initialize(self, language):
        spanish_lower = np.matrix(
            [
                [u"º", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"'", u"¡"],
                [None, u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", u"`", u"+"],
                [None, u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u"ñ", u"´", u"ç"],
                [u"<", u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"-", None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None],
            ],
            dtype="U",
        )
        spanish_upper = np.matrix(
            [
                [u"ª", u"!", u'"', u"·", u"$", u"%", u"&", u"/", u"(", u")", u"=", u"?", u"¿"],
                [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"^", u"*"],
                [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u"Ñ", u"¨", u"Ç"],
                [u">", u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u";", u":", u"_", None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None],
            ],
            dtype="U",
        )

        english_lower = np.matrix(
            [
                [u"`", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"-", u"=", None],
                [None, u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", "[", "]", "\\"],
                [None, u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u";", u"'", None, None],
                [None, u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"/", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )
        english_upper = np.matrix(
            [
                [u"~", u"!", u"@", u"#", u"$", u"%", u"^", u"&", u"*", u"(", u")", u"_", u"+", None],
                [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"{", u"}", u"|"],
                [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u":", u'"', None, None],
                [None, u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u"<", u">", u"?", None, None, None],
                [None, None, None, " ", " ", " ", " ", " ", None, None, None, None, None, None],
            ],
            dtype="U",
        )

        if language.lower() == "english":
            self.qwerty_grid = english_lower
            self.QWERTY_grid = english_upper
        elif language.lower() == "spanish":
            self.qwerty_grid = spanish_lower
            self.QWERTY_grid = spanish_upper
        else:
            raise ValueError("The language " + language + " is not supported yet!")

        self.average_x, self.average_y = self.qwerty_grid.shape[1] / 2, self.qwerty_grid.shape[0] / 2
        logger.debug("Average locations are : " + str(self.average_x) + ", " + str(self.average_y))

    def get_location(self, first_char, second_char):
        # only normalize if character doesn't exist in keyboard
        if first_char not in self.qwerty_grid and first_char not in self.QWERTY_grid:
            first_char = helper.normalize(first_char)
        if second_char not in self.qwerty_grid and second_char not in self.QWERTY_grid:
            second_char = helper.normalize(second_char)

        loc1 = np.where(self.qwerty_grid == first_char)
        # if we can't find it, look in other map
        if len(loc1[0]) == 0:
            loc1 = np.where(self.QWERTY_grid == first_char)
        # still can't find it, just return average
        if len(loc1[0]) == 0:
            logger.debug("Couldn't find first character " + first_char + " so returning average locations")
            return self.average_x, self.average_y
        loc1_row = loc1[0][0]
        loc1_column = loc1[1][0]
        loc2 = np.where(self.qwerty_grid == second_char)
        # if we can't find it, look in other map
        if len(loc2[0]) == 0:
            loc2 = np.where(self.QWERTY_grid == second_char)
        # still can't find it, just return average
        if len(loc2[0]) == 0:
            logger.debug("Couldn't find second character " + second_char + " so returning average locations")
            return self.average_x, self.average_y
        loc2_row = loc2[0][0]
        loc2_column = loc2[1][0]

        # Handle spacebar case
        if first_char == " ":
            if 3 <= loc2_column <= 7:
                loc1_column = loc2_column
            elif loc2_column < 3:
                loc1_column = 3
            elif loc2_column > 7:
                loc1_column = 7

        if second_char == " ":
            if 3 <= loc1_column <= 7:
                loc2_column = loc1_column
            elif loc1_column < 3:
                loc2_column = 3
            elif loc1_column > 7:
                loc2_column = 7
        x = loc1_row - loc2_row
        y = loc1_column - loc2_column
        return x, y

    def calculate_distance(self, first_char, second_char):
        logger.debug("calculate distance for %s; %s" % (first_char, second_char))
        if first_char == second_char:
            logger.debug("same")
            dist = 0
        else:
            x, y = self.get_location(first_char, second_char)
            if x == 0 and y == 0:
                logger.debug("same")
                dist = 0
            elif (x == 0 and (np.abs(y) <= 1)) or (np.abs(x) <= 1 and y == 0):
                logger.debug("orthogonal")
                dist = 1
            elif (x == -1 and y == 1) or (x == 1 and y == -1):
                logger.debug("diagonal")
                dist = 1
            else:
                logger.debug("other")
                dist = np.sqrt(np.square(x) + np.square(y))
        return dist

    def get_hand(self, char):
        char = helper.normalize(char)
        loc = np.where(self.qwerty_grid == char)
        # if we can't find it, look in other map
        if len(loc[0]) == 0:
            loc = np.where(self.QWERTY_grid == char)
        # still can't find it, just return none
        if len(loc[0]) == 0:
            logger.debug("Couldn't find  character " + char + " so returning none")
            return None
        loc_row = loc[0][0]
        loc_column = loc[1][0]

        # special case for spacebar, can be typed with either hand
        if loc_row == 4:
            return "s"
        else:
            if loc_column <= 5:
                return "l"
            else:
                return "r"

    def same_hand(self, first_char, second_char):
        hand_first = self.get_hand(first_char)
        hand_second = self.get_hand(second_char)
        if hand_first is None or hand_second is None:
            return False
        elif hand_first == "s" or hand_second == "s":
            return True
        else:
            return hand_first == hand_second
