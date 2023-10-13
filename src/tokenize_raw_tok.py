#!/usr/bin/env python
# -*- coding: utf8 -*-


if __name__ == '__main__':


    from nltk.tokenize import word_tokenize
    #input_file_path = '../data/3.0.0/en/gold_silver/train.txt.raw'
    input_file_path = '../DRS_parsing/data/pmb-3.0.0/gold/train.txt.raw'


    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        # Read the content of the file
        input_text = input_file.read()

    res = []

    for sent in input_text.split('\n'):
        res.append(" ".join(word_tokenize(sent)))


    with open(input_file_path + "_compare.tok", 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(res))


