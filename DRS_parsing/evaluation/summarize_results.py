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

def fancyPrint(res_list, full_res):
    if not res_list is None and not len(res_list) == 0:
        print("-------- F-Score ---------")
        print("F-score, DRS only: {0} {{{1}}}".format(round(res_list[0], 4), ', '.join(map(str, [res[0] for res in full_res]))))
        print("F-score, alignment also correct : {0} {{{1}}}".format(round(res_list[1], 4), ', '.join(map(str, [res[1] for res in full_res]))))
        print("-------- Alignment Accuracy ---------")
        print("Alignment Accuracy: {0} {{{1}}}".format(round(res_list[4], 3), ', '.join(map(str, [res[4] for res in full_res]))))
        print("Alignment Accuracy token for fully correct DRS: {0} {{{1}}}".format(round(res_list[5], 3), ', '.join(map(str, [res[5] for res in full_res]))))
        print("-------- Counts ---------")
        print("Fully Correct DRS: {0} out of {1} ({2}%)".format(round(res_list[13], 3), round(res_list[14], 3), round(res_list[13] / res_list[14]* 100), 3 ) )
        print("Count Alignment Errors: {0}".format(round(res_list[10], 3)))


def sum_lists_elementwise(lists, num_runs):
    # Ensure all lists have the same length
    if not all(len(lst) == len(lists[0]) for lst in lists[1:]):
        raise ValueError("All lists must have the same length for element-wise addition")

    # Use zip to iterate over elements at the same index and sum them up
    result_sum = [sum(nums)/num_runs for nums in zip(*lists)]

    return result_sum

def average_fine_tuned(tune_dir, calc_from_output = False):
    test_f1 = 0
    dev_f1 = 0
    dev_num = 0
    test_num = 0
    dev_results = []
    test_results = []
    for run in os.listdir(tune_dir):
        run_path = os.path.join(tune_dir, run)
        if os.path.isdir(run_path):
            if calc_from_output:
                output_path = os.path.join(run_path, "output")
                if os.path.isdir(output_path):
                    dev_file = os.path.join(output_path, "output_{}_epoch_{}.seq.drs.attentAlign.out".format("dev", epochs))
                    test_file = os.path.join(output_path, "output_{}_epoch_{}.seq.drs.attentAlign.out".format("test", epochs))
                    dev_present = (os.path.exists(dev_file) and os.path.isfile(dev_file))
                    test_present = (os.path.exists(test_file) and os.path.isfile(test_file))
                    if (dev_present):
                        args_str = '-f1 {} -f2 {} -g {}'.format(dev_file, os.path.join(d2, "dev.txt"), d3)
                        #if "tok" in args_str and "nontok" not in args_str:
                        args_str = args_str + add_argstring
                        args = counter.build_arg_parser(args_str)
                        #dev_f1 += counter.main(args, verbose=False)[2]
                        dev_results.append(counter.main(args, verbose=False))
                        dev_num += 1
                    if (test_present):
                        args_str = '-f1 {} -f2 {} -g {}'.format(test_file, os.path.join(d2, "test.txt"), d3)
                        #if "tok" in args_str and "nontok" not in args_str:
                        args_str = args_str + add_argstring
                        args = counter.build_arg_parser(args_str)
                        #test_f1 += counter.main(args, verbose=False)[2]
                        test_results.append(counter.main(args, verbose=False))
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
    print(dev_results)
    print(test_results)
    return dev_results, dev_num, test_results, test_num

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

    #add_argstring = " -rt -et -ei -ti -ae //home/krise/Documents/masterarbeit/experiment_results/secondnew_tok/run1/alignmenterrors.txt -ic"
    add_argstring = " -rt -et -ei -ti -ae /home/krise/Documents/masterarbeit/experiment_results/tok_bilinearAtt_lstm_4epoch_refsep_06_02_24_cpy/run1/alignmenterrors.txt"

    print(add_argstring)

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
            dev_results, num_dev, test_results, num_test = average_fine_tuned(finetuned_dir)
            finetune_dev = sum_lists_elementwise(dev_results, num_dev)
            finetune_test = sum_lists_elementwise(test_results, num_test)
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

        if fine_tuned_present:
            dev_results, num_dev, test_results, num_test =average_fine_tuned(finetuned_dir, calc_from_output = True)
            finetune_dev = sum_lists_elementwise(dev_results, num_dev)
            finetune_test = sum_lists_elementwise(test_results, num_test)
            print("Dev-Set:")
            fancyPrint(finetune_dev, dev_results)
            print("\n")
            print("Test-Set")
            fancyPrint(finetune_test, test_results)

