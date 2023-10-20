#!/usr/bin/env python
# -*- coding: utf8 -*-
import argparse
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_file", default='', type=str, help="input-file that is to be tokenized")

    args = parser.parse_args()
    input_file_path = args.input_file

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()
    res = []

    for sent in input_text.split('\n'):
        res.append(" ".join(word_tokenize(sent)))

    with open(input_file_path + ".tok" , 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(res))


