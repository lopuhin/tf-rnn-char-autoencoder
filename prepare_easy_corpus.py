#!/usr/bin/env python
# encoding: utf-8

import re
import codecs
import argparse
from collections import Counter
from operator import itemgetter

import numpy as np


WORDS_RE = re.compile(r'\w+', re.U)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--fn', default='min', help='min or mean')
    args = parser.parse_args()
    with codecs.open(args.filename, 'rb', 'utf-8') as f:
        freqs = Counter(word for line in f
                        for word in WORDS_RE.findall(line))
        f.seek(0)
        fn = getattr(np, args.fn)
        weights = [(i, line_weight(line, freqs, fn))
                   for i, line in enumerate(f)]
        weights.sort(key=itemgetter(1), reverse=True)
        f.seek(0)
        easy = set(map(itemgetter(0), weights[:args.n]))
        for i, line in enumerate(f):
            if i in easy:
                print line.encode('utf-8'),


def line_weight(line, freqs, fn):
    return fn([freqs[w] for w in WORDS_RE.findall(line)] or [0])


if __name__ == '__main__':
    main()
