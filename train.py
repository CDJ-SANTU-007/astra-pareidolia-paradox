"""Fit the Astra classifier on the supplied training dataset."""

import sys

from classifier import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
