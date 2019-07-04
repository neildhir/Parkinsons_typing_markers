"""
-Paper: https://arxiv.org/pdf/1602.02410.pdf
-Implementation borrows from: https://github.com/offbit/char-models
-As well as: https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10
"""


# Misc
import re
import pandas as pd
import numpy as np
import sklearn

# For splitting data
from sklearn.model_selection import train_test_split

# Preprocessing
import nltk
from nltk.tokenize import sent_tokenize

# Modelling
import keras
