#!/usr/bin/env python

"""
Command line script for preprocessing execution.
"""
from src.preprocess import preprocessMRC


def main():

    proc = preprocessMRC()
    out = proc()
    out.to_csv("data/MRC/preproc/EnglishData-preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
