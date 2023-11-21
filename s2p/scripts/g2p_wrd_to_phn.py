#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from collections import defaultdict
from g2p_en import G2p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        action="store_true",
        help="if set, compacts phones",
    )
    parser.add_argument(
        "--only_phonemes",
        action="store_true",
        help="if set, filter out non-phone set phones",
    )
    args = parser.parse_args()

    compact = args.compact
    only_phonemes = args.only_phonemes

    wrd_to_phn = defaultdict(lambda: False)
    g2p = G2p()
    for line in sys.stdin:
        words = line.strip().split()
        phones = []
        valid_phonemes = [p for p in g2p.phonemes]
        for p in g2p.phonemes:
            if p[-1].isnumeric():
                valid_phonemes.append(p[:-1])
        for w in words:
            if not wrd_to_phn[w]:
                phns = g2p(w)
                if compact:
                    phns = [
                        p[:-1] if p[-1].isnumeric() else p for p in phns
                    ]
                if only_phonemes:
                    phns = list(filter(lambda x: x in valid_phonemes, phns))
                    # print(phns)
                wrd_to_phn[w] = phns
            phones.extend(wrd_to_phn[w])
        try:
            print(" ".join(phones))
        except:
            print(wrd_to_phn, words, phones, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
