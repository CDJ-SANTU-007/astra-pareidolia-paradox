"""Generate predictions for the complete evaluation batch."""

import sys

from classifier import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
