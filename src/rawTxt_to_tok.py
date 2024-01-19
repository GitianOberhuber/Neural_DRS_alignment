#!/usr/bin/env python
# -*- coding: utf8 -*-
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_file", default='', type=str, help="input-file that is to be tokenized")

    args = parser.parse_args()
    input_file_path = args.input_file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()

    txt = re.split(r'%%%\s*', input_text)
    txt = txt[1:] #remove first \n resulting from splitting
    txt = txt[2::3] #keep every third line starting with '%%%%', which corresponds to the nat. lang. utterance followed by DRS
    txt = [x.split("\n")[0].replace('ø ', '') for x in txt] #discarding DRS
    txt[-1] = txt[-1] + "\n"

    with open(input_file_path + ".raw.tok" , 'w') as output_file:
        output_file.write('\n'.join(txt))