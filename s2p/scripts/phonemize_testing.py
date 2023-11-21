#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import sys
import string


def get_parser():
    parser = argparse.ArgumentParser(
        description="converts words to phones adding optional silences around in between words"
    )
    parser.add_argument(
        "--lexicon",
        help="lexicon to convert to phones",
        required=True,
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    sil = "<SIL>"
    unk = "<UNK>"

    wrd_to_phn = {}

    with open(args.lexicon, "r") as lf:
        for line in lf:
            items = line.rstrip().split()
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]

    for line in sys.stdin:
        line = line.translate(str.maketrans('', '', ''.join(string.punctuation.split('\''))))
        words = line.strip().split()

        # if not all(w in wrd_to_phn for w in words):
        #     continue

        phones = []
        # if surround:
        #     phones.append(sil)

        for i, w in enumerate(words):
            if w not in wrd_to_phn:
                phones.append(unk)
                continue
            phones.extend(wrd_to_phn[w])

        print(" ".join(phones))


if __name__ == "__main__":
    main()
