#!/usr/bin/env python

"""
Command line script for preprocessing execution

"""
from src.preprocess import preprocessMRC


def main():

    proc = preprocessMRC()
    i = 'script'
    out = proc()
    out.to_csv("data/MRC/preproc/char_time/EnglishData-preprocessed_attempt_{}.csv".format(i), index=False)



if __name__ == '__main__':
     main()