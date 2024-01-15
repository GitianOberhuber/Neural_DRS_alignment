#!/usr/bin/env python
# -*- coding: utf8 -*-
import argparse
import os
import re
from DRS_parsing.evaluation import counter

def read_f1_from_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("F-score"):
                return line.strip()

    return None

def average_fine_tuned(tune_dir, calc_from_output = False):
    test_f1 = 0
    dev_f1 = 0
    dev_num = 0
    test_num = 0
    for run in os.listdir(tune_dir):
        run_path = os.path.join(tune_dir, run)
        if os.path.isdir(run_path):
            if calc_from_output:
                output_path = os.path.join(run_path, "output")
                if os.path.isdir(output_path):
                    dev_file = os.path.join(output_path, "output_{}_epoch_{}.seq.drs.out".format("dev", epochs))
                    test_file = os.path.join(output_path, "output_{}_epoch_{}.seq.drs.out".format("test", epochs))
                    dev_present = (os.path.exists(dev_file) and os.path.isfile(dev_file))
                    test_present = (os.path.exists(test_file) and os.path.isfile(test_file))
                    if (dev_present):
                        args_str = '-f1 {} -f2 {} -g {}'.format(dev_file, os.path.join(d2, "dev.txt"), d3)
                        if "tok" in args_str and "nontok" not in args_str:
                            args_str = args_str + " -rt -et -ei"
                        args = counter.build_arg_parser(args_str)
                        dev_f1 += counter.main(args, verbose=False)[2]
                        dev_num += 1
                    if (test_present):
                        args_str = '-f1 {} -f2 {} -g {}'.format(test_file, os.path.join(d2, "test.txt"), d3)
                        if "tok" in args_str and "nontok" not in args_str:
                            args_str = args_str + " -rt -et -ei"
                        args = counter.build_arg_parser(args_str)
                        test_f1 += counter.main(args, verbose=False)[2]
                        test_num += 1
            else:
                eval_path = os.path.join(run_path, "eval")
                if os.path.isdir(eval_path):
                    dev_file = os.path.join(eval_path, "dev_eval.txt")
                    test_file = os.path.join(eval_path, "test_eval.txt")
                    dev_present = (os.path.exists(dev_file) and os.path.isfile(dev_file))
                    test_present = (os.path.exists(test_file) and os.path.isfile(test_file))
                    if (dev_present):
                        dev_f1 += float(re.sub(r'F-score\s*:\s*', '', read_f1_from_file(dev_file)))
                        dev_num +=1
                    if (test_present):
                        test_f1 += float(re.sub(r'F-score\s*:\s*', '', read_f1_from_file(test_file)))
                        test_num +=1
    return dev_f1/dev_num, test_f1/test_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", default='', type=str, help="directory containing experiment output in form of eval metrices in a text file")
    parser.add_argument('-d1', "--predictions", default='', type=str, help="directory containing predicted DRSs")
    parser.add_argument('-d2', "--goldlabels", default='', type=str, help="directory containing goldlabel DRSs")
    parser.add_argument('-si', "--clf_signature", default='', type=str, help="path to clf signature")
    parser.add_argument('-e', "--number_epochs", default='3', type=str, help="the number of epochs the experiment ran for")
    parser.add_argument('-c', "--calcArgs", default='', type=str, help="additional arguments to pass to counter")


    args = parser.parse_args()
    exp_dir = args.directory
    d1 = args.predictions
    d2 = args.goldlabels
    d3 = args.clf_signature
    epochs = args.number_epochs

    if (exp_dir == "" and d1 == "" and d2 == "") or (not exp_dir == "" and not (d1 == "" and d2 == "")):
        print("Either -d or -d1 + -d2 options must be specified!")
        exit(-1)


    if exp_dir != "":
        if not(os.path.exists(exp_dir) and os.path.isdir(exp_dir)):
            print("Experiment directory could not be found!")
            exit(-1)

        eval_dir = os.path.join(exp_dir, "eval")
        if not(eval_dir and eval_dir):
            print("eval directory could not be found inside experiment directory!")
            exit(-1)

        dev_file = os.path.join(eval_dir, "dev_eval.txt")
        test_file = os.path.join(eval_dir, "test_eval.txt")

        dev_present = (os.path.exists(dev_file) and os.path.isfile(dev_file))
        test_present = (os.path.exists(test_file) and os.path.isfile(test_file))

        if not (dev_present or test_present):
            print("eval directory contains neither a dev file nor a test file!")
            exit(-1)

        finetuned_dir = os.path.join(exp_dir, "fine-tuned")
        fine_tuned_present = os.path.exists(finetuned_dir) and os.path.isdir(finetuned_dir)

        if dev_present:
            print("Non-fine-tuned, dev-set:")
            print(read_f1_from_file(dev_file))

        if test_present:
            print("Non-fine-tuned, test-set:")
            print(read_f1_from_file(test_file))

        if fine_tuned_present:
            finetune_dev, finetune_test = average_fine_tuned(finetuned_dir)
            print("Fine-tuned, averaged, dev-set:")
            print(round(finetune_dev, 3))
            print("Fine-tuned, averaged, test-set:")
            print(round(finetune_test,3))

    else:
        if not (os.path.exists(d1) and os.path.isdir(d1)) or not (os.path.exists(d2) and os.path.isdir(d2)):
            print("Experiment directories could not be found!")
            exit(-1)

        output_dir = os.path.join(d1, "output")
        if not (output_dir and output_dir):
            print("output directory could not be found inside experiment directory!")
            exit(-1)

        dev_file = os.path.join(output_dir, "output_{}_epoch_{}.seq.drs.out".format("dev", epochs))
        test_file = os.path.join(output_dir, "output_{}_epoch_{}.seq.drs.out".format("test", epochs))

        dev_present = (os.path.exists(dev_file) and os.path.isfile(dev_file))
        test_present = (os.path.exists(test_file) and os.path.isfile(test_file))

        if not (dev_present or test_present):
            print("eval directory contains neither a dev file nor a test file!")
            exit(-1)

        finetuned_dir = os.path.join(d1, "fine-tuned")
        fine_tuned_present = os.path.exists(finetuned_dir) and os.path.isdir(finetuned_dir)

        if dev_present:
            args_str = '-f1 {} -f2 {} -g {}'.format(dev_file, os.path.join(d2, "dev.txt"), d3)
            if "tok" in args_str and "nontok" not in args_str:
                args_str = args_str + " -rt -et -ei"
            args = counter.build_arg_parser(args_str)
            print("Non-fine-tuned, dev-set:")
            print(counter.main(args, verbose= False))

        if test_present:
            args_str = '-f1 {} -f2 {} -g {} -rt'.format(test_file, os.path.join(d2, "test.txt"), d3)
            if "tok" in args_str and "nontok" not in args_str:
                args_str = args_str + " -rt -et -ei"
            args = counter.build_arg_parser(args_str)
            print("Non-fine-tuned, test-set:")
            print(counter.main(args, verbose= False))

        if fine_tuned_present:
            finetune_dev, finetune_test = average_fine_tuned(finetuned_dir, calc_from_output = True)
            print("Fine-tuned, averaged, dev-set:")
            print(round(finetune_dev, 3))
            print("Fine-tuned, averaged, test-set:")
            print(round(finetune_test, 3))
