#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes an F-score between two DRSs. It is based on and very similar to SMATCH.
For detailed description of smatch, see http://www.isi.edu/natural-language/drs/smatch-13.pdf

As opposed to AMRs, this script takes the clauses directly as input. Each clause should be on a
single line, with DRSs separated by a newline. Lines starting with '%' are ignored.

Command line options:

-f1   : First file with DRS clauses, usually produced file
-f2   : Second file with DRS clauses, usually gold file
-r    : Number of restarts used (default 20)
-p    : Number of parallel threads to use (default 1)
-mem  : Memory limit per parallel thread (default 1G)
-s    : What kind of smart initial mapping we use:
	  -no    No smart mappings
	  -conc  Smart mapping based on matching concepts (their match is likely to be in the optimal mapping)
-runs : Number of runs to average over, if you want a more reliable result (there is randomness involved in the initial restarts)
-prin : Print more specific output, such as individual (average) F-scores for the smart initial mappings, and the matching and non-matching clauses
-sig  : Number of significant digits to output (default 4)
-b    : Use this for baseline experiments, comparing a single DRS to a list of DRSs. Produced DRS file should contain a single DRS.
-ms   : Instead of averaging the score, output a score for each DRS
-ms_file: Print the individual scores to a file (one score per line)
-nm   : If added, do not print the clause mapping
-pa   : Partial matching of clauses instead of full matching -- experimental setting!
		Means that the clauses x1 work "n.01" x2 and x1 run "n.01" x2 are able to match for 0.75, for example
-st   : Printing statistics regarding number of variables and clauses, don't do matching
-ic   : Include REF clauses when matching (otherwise they are ignored because they inflate the matching)
-ds   : Print detailed stats about individual clauses, parameter value is the minimum count of the clause to be included in the printing
-m    : Max number of clauses for a DRS to still take them into account - default 0 means no limit
-g    : Path of the file that contains the signature of clausal forms
-dse  : Change all senses to a default sense
-dr   : Change all roles to a default role
-dc   : Change all concepts to a default concept
-ill  : What to do with ill-formed DRSs. Throw an error (default), input dummy/SPAR DRS or try to output a score anyway (unofficial!)
-coda : For the CodaLab usage. Given a 'name', the script creates 'name.txt' and 'name.html' files
"""

import os
import random
import time
import argparse
import re
import sys
import multiprocessing
from multiprocessing import Pool
import json #reading in dict

from src.uts import drs_string_to_list, filter_out_doubleAlignment

try:
	import cPickle as pickle
except ImportError:
	import pickle

from numpy import median

#try:
# only needed for Python 2
#	reload(sys)
#	sys.setdefaultencoding('utf-8')  # necessary to avoid unicode errors
#except:
#	pass

import psutil  # for memory usage

# Imports for the hillclimbing algorithm
from hill_climbing import *
# Imports for format checking
from clf_referee import check_clf
from clf_referee import get_signature
# import html priting for codalab
from html_results import coda_html
# Import utils
from utils_counter import *
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import editdistance


def build_arg_parser(args = None):
	parser = argparse.ArgumentParser(description="Counter calculator -- arguments")
	# Main arguments
	parser.add_argument('-f1', required=True, type=str,
						help='First file with DRS clauses, DRSs need to be separated by blank line')
	parser.add_argument('-f2', required=True, type=str,
						help='Second file with DRS clauses, DRSs need to be separated by blank line')

	# Optimization (memory, speed, restarts, initial mappings)
	parser.add_argument('-r', '--restarts', type=int,
						default=20, help='Restart number (default: 20)')
	parser.add_argument('-p', '--parallel', type=int, default=1,
						help='Number of parallel threads we use (default 1)')
	parser.add_argument('-mem', '--mem_limit', type=int, default=1000,
						help='Memory limit in MBs (default 1000 -> 1G). Note that this is per parallel thread! If you use -par 4, each thread gets 1000 MB with default settings.')
	parser.add_argument('-s', '--smart', default='conc', action='store', choices=[
		'no', 'conc'], help='What kind of smart mapping do we use (default concepts)')

	# Output settings (often not necessary to add or change), for example printing specific stats to a file, or to the screen
	parser.add_argument('-prin', action='store_true',
						help='Print very specific output - matching and non-matching clauses and specific F-scores for smart mappings')
	parser.add_argument('-ms', action='store_true', default=False,
						help='Output multiple scores (one pair per score) instead of a single document-level score (Default: false)')
	parser.add_argument('-ms_file', default = '',
						help='The file where we print the individual scores per DRS to -- one score per line (float) -- default empty means do not print to file')
	parser.add_argument('-al', '--all_idv', action='store_true',
						help='Add all idv information in the --ms_file file (match, prod, gold), not just the F-score')
	parser.add_argument('-sig', '--significant', type=int,
						default=4, help='significant digits to output (default: 4)')
	parser.add_argument('-st', '--stats', default='', type=str,
						help='If added this is the file we print a pickled dictionary to with statistics about number of clauses and variables.')
	parser.add_argument('-ds', '--detailed_stats', type = int, default = 0,
						help='If we add a value > 0 we print statistics about individual types of clauses that match or do not match, e.g. how well do we do on producing Theme, default 0 means do nothing')
	parser.add_argument('-nm', '--no_mapping', action='store_true',
						help='If added, do not print the mapping of variables in terms of clauses')
	parser.add_argument('-g', '--signature', dest='sig_file', default = '',
						help='If added, this contains a file with all allowed roles otherwise a simple signature is used that\
              				  mainly recognizes operators based on their formatting')
	parser.add_argument('-coda', '--codalab', default='',
						help='a filename for which evaluation results for CodaLab are written in filename.txt and filename.html (default no writing)')

	# Experiments with changing the input/output, or the matching algorithm
	# If you add this the results will differ from the general F-score
	parser.add_argument('-ill', default = 'error', choices =['error','dummy','spar', 'score'],
						help='What to do when encountering an ill-formed DRS. Throw an error (default), input dummy or spar DRS, or give a score anyway (those scores are not official though!)')
	parser.add_argument('-runs', type=int, default=1,
						help='Usually we do 1 run, only for experiments we can increase the number of runs to get a better average')
	parser.add_argument('-m', '--max_clauses', type=int, default=0,
						help='Maximum number of clauses for DRS (default 0 means no limit)')
	parser.add_argument('-b', '--baseline', action='store_true', default=False,
						help="Helps in deciding a good baseline DRS. If added, prod-file must be a single DRS, gold file a number of DRSs to compare to (default false)")
	parser.add_argument('-pa', '--partial', action='store_true',
						help='Do partial matching for the DRSs (experimental!)')
	parser.add_argument('-dse', '--default_sense', action='store_true',
						help='Add a default sense for all word-senses (exclude effect of getting sense correct)')
	parser.add_argument('-dr', '--default_role', action='store_true',
						help='Add a default role for all role clauses (exclude effect of getting role correct)')
	parser.add_argument('-dc', '--default_concept', action='store_true',
						help='Add a default concept + sense for all concept clauses (exclude effect of getting concepts correct)')
	parser.add_argument('-ic', '--include_ref', action='store_true',
						help='Include REF clauses when matching -- will inflate the scores')
	parser.add_argument("-rt", "--reference_input_token", action='store_true',
						help="Keep the references linking DRS clauses to a token in the natural language input instead of discarding them")
	parser.add_argument("-et", "--evaluate_token", action='store_true',
						help="When comparing DRS clauses, take references to original token into account and only consider a clause pair to be matching"
							 "if their token references also match")
	parser.add_argument("-ei", "--evaluate_indices", action='store_true',
						help="When comparing DRS clauses, take references to original token start- and end-index into account and only consider a clause pair to be matching"
							 "if their token indices also match")
	parser.add_argument("-ti", "--evaluate_token_ignore_casing", action='store_true',
						help="When -ti is specified, do not take casing into account when evaluating alignment")
	parser.add_argument("-mr", "--match_roughly", action='store_true',
						help="When -mr is specified, consider the tokens of the alignment to be matching if they are similar, i.e one is a lemma, synonym or stemmed version of the other")
	parser.add_argument('-ae', "--alignment_errors", required=False, type=str,
						help='Path to a textfile that will be filled with DRS that contain incorrect alignments, only valid if -rt and -et are also specified')

	if args is not None:
		args = args.split()
		args = parser.parse_args(args)
	else:
		args = parser.parse_args()

	# Check if files exist
	if not os.path.exists(args.f1):
		raise ValueError("File for -f1 does not exist")
	if not os.path.exists(args.f2):
		raise ValueError("File for -f2 does not exist")

	# Check if combination of arguments is valid
	if args.ms and args.runs > 1:
		raise NotImplementedError("Not implemented to average over individual scores, only use -ms when doing a single run")
	if args.restarts < 1:
		raise ValueError('Number of restarts must be larger than 0')

	if args.ms and args.parallel > 1:
		print('WARNING: using -ms and -p > 1 messes up printing to screen - not recommended')
		time.sleep(5)  # so people can still read the warning

	if args.ill in ['dummy', 'spar']:
		print('WARNING: by using -ill {0}, ill-formed DRSs are replaced by a {0} DRS'.format(args.ill))
		time.sleep(3)
	elif args.ill == 'score':
		print ('WARNING: ill-formed DRSs are given a score as if they were valid -- results in unofficial F-scores')
		time.sleep(3)

	if args.runs > 1 and args.prin:
		print('WARNING: we do not print specific information (-prin) for runs > 1, only final averages')
		time.sleep(5)

	if args.partial:
		raise NotImplementedError('Partial matching currently does not work')
	return args


def remove_refs_invalid_drs(clause_list):
	res = []
	for i,drs in enumerate(clause_list):
		invalid_drs = False
		for cl in drs:
			for el in cl:
				if "alwayswrongconcept" in el:
					invalid_drs = True
		if invalid_drs:
			drs_without_ref = []
			for cl in drs:
				if cl[1] == "REF":
					pass
				else:
					drs_without_ref.append(cl)
			res.append(drs_without_ref)
		else:
			res.append(drs)
	return res

def remove_refs(clause_list, original_clauses, rt = False):
	'''Remove unneccessary/redundant b REF x clauses, only keep them if x never occurs again in the same box'''
	final_clauses, final_original = [], []
	for tup_idx, tup_set in enumerate(clause_list):
		cur_tup = []
		cur_orig = []
		for idx, spl_tup in enumerate(tup_set):
			if spl_tup[1] == 'REF':
				if not var_occurs(tup_set, spl_tup[2], spl_tup[0], idx, rt):  # only add if the variable does not occur afterwards
					cur_tup.append(spl_tup)
					cur_orig.append(original_clauses[tup_idx][idx])
			else:
				cur_tup.append(spl_tup)
				cur_orig.append(original_clauses[tup_idx][idx])
		final_clauses.append(cur_tup)
		final_original.append(cur_orig)
	return final_clauses, final_original


def get_clauses(file_name, signature, ill_type, rt = False, include_REF = False, read_nl_sentences = False):
	'''Function that returns a list of DRSs (that consists of clauses)'''
	clause_list, original_clauses, cur_orig, cur_clauses, input_sequences = [], [], [], [], []

	with open(file_name, 'r') as in_f:
		input_lines = in_f.read().split('\n')
		count_percent_sign = 0 #for reading the natural language sentence, i.e every third comment (indicated by %),
		for idx, line in enumerate(input_lines):
			if line.strip().startswith('%'):
				if read_nl_sentences and line.strip().startswith('%%%'):
					count_percent_sign += 1
					if count_percent_sign % 3 == 0: #read every third comment instead of discarding
						input_sequence = line.replace("%%% ", "").replace("ø ", "").replace("~", " ")
						count_percent_sign = 0
				else:
					pass

			elif not line.strip():
				if cur_clauses:  # newline, so DRS is finished, add to list. Ignore double/clause newlines
					# First check if the DRS is valid, will error if invalid
					try:
						#f = drs_string_to_list(cur_clauses, rt)
						check_clf([tuple(c) for c in cur_clauses], signature, v=False, alignment= rt)
						clause_list.append(cur_clauses)
						original_clauses.append(cur_orig)
						if read_nl_sentences:
							input_sequences.append(input_sequence)
					except Exception as e:
						if ill_type == 'error':
							raise ValueError(e)
						elif ill_type == 'dummy':
							# FIXME: uncomment
							print('WARNING: DRS {0} is ill-formed and replaced by a dummy DRS'.format(len(clause_list) +1))
							clause_list.append(dummy_drs())
							original_clauses.append([" ".join(x) for x in dummy_drs()])
						elif ill_type == 'spar':
							print('WARNING: DRS {0} is ill-formed and replaced by the SPAR DRS'.format(len(clause_list) +1))
							clause_list.append(spar_drs())
							original_clauses.append([" ".join(x) for x in spar_drs()])
						elif ill_type == 'score':
							print('WARNING: DRS {0} is ill-formed, but try to give a score anyway - might still error later'.format(len(clause_list) +1))

							clause_list.append(cur_clauses)
							original_clauses.append(cur_orig)
				cur_clauses = []
				cur_orig = []
			else:
				if rt:
					tmp = drs_string_to_list([line], rt)
					tmp = filter_out_doubleAlignment(tmp)
					tmp[0].insert(-3, '%')
					cur_clauses.append(tmp[0])

				else:
					cur_clauses.append(line.split(' %', 1)[0].strip().split()) #remove comments
				#ls = line.split(' %', 1)[0].strip().split()
				#if len(ls) > 4:
				#	if len(ls) == 5:
				#			ls = ls[:3]
				#	else:
				#		ls = ls[:4]
				#cur_clauses.append(ls)
				cur_orig.append(line)

	if cur_clauses:  # no newline at the end, still add the DRS
		clause_list.append(cur_clauses)
		original_clauses.append(cur_orig)

	# Invert -of relations and reorder inv_boxes if they contain a constant between quotes
	inv_boxes = DRS(signature).inv_boxes
	for drs in clause_list:
		for clause in drs:
			if rt:
				if len(clause) == 7 and is_role(clause[1]) and clause[1].endswith('Of') and len(clause[1]) > 2:
					clause[2], clause[3] = clause[3], clause[2]
					clause[1] = clause[1][:-2]
				elif clause[1] in inv_boxes and len(clause) == 7 and between_quotes(clause[2]) and not between_quotes(clause[3]):
					clause[2], clause[3] = clause[3], clause[2]
			else:
				if len(clause) == 4 and is_role(clause[1]) and clause[1].endswith('Of') and len(clause[1]) > 2:
					# Switch clauses and remove the -Of
					clause[2], clause[3] = clause[3], clause[2]
					clause[1] = clause[1][:-2]
				elif clause[1] in inv_boxes and len(clause) == 4 and between_quotes(clause[2]) and not between_quotes(clause[3]):
					# b1 NEQ x1 x2 is equal to b1 NEQ x2 x1
					# If one of the two arguments is between quotes, rewrite them in such a way
					# that it can always match
					# For example rewrite b1 NEQ "speaker" x1 to b1 NEQ x1 "speaker"
					# If there are two variables or two items between quotes, do nothing
					clause[2], clause[3] = clause[3], clause[2]

	# If we want to include REF clauses we are done now
	if include_REF:
		return remove_refs_invalid_drs(clause_list), original_clauses, input_sequences
	else: #else remove redundant REF clauses
		final_clauses, final_original = remove_refs(clause_list, original_clauses, rt)
		return final_clauses, final_original, input_sequences


def var_occurs(clauses, var, box, idx, rt = False):
	'''Check if variable occurs with same box in one of the next clauses'''
	for cur_idx in range(0, len(clauses)):
		spl_tup = clauses[cur_idx]
		if rt:
			if '%' in spl_tup:
				tmp_spl_tup = [x for x in spl_tup if x != '%']
			if len(tmp_spl_tup) == 6:
				spl_tup = tmp_spl_tup[:3]
			elif len(tmp_spl_tup) == 7:
				spl_tup = tmp_spl_tup[:4]
		if cur_idx != idx and spl_tup[0] == box and (var == spl_tup[-1] or var == spl_tup[-2]):
			return True
	return False


def rewrite_concept(concept, sense, en_sense_dict):
	'''Rewrite concepts to their WordNet 3.0 synset IDs'''
	fixed_sense = sense.replace('"', '') #replace quotes
	dict_key = concept + '.' + fixed_sense
	if dict_key not in en_sense_dict: #not in WordNet, return original values
		return concept, sense, []
	else:
		conc_sense = en_sense_dict[dict_key]
		# Now divide back again in concept + sense with quotes
		new_concept = ".".join(conc_sense.split('.')[0:-2])
		new_sense = '"' + ".".join(conc_sense.split('.')[-2:]) + '"'
		if dict_key == conc_sense: #nothing normalized after all
			return new_concept, new_sense, []
		else:
			return new_concept, new_sense, [dict_key, conc_sense]


def fill_baseline_list(baseline_drs, clauses_gold_list):
	'''Fill baseline drss so that we can compare all DRSs to a baseline'''
	return [baseline_drs for _ in range(len(clauses_gold_list))]


def get_clause_id(clause):
	'''Get individual clause ID, e.g. Theme, work, IMP
	   However, for concepts we also want the sense!'''
	spl_line = clause.split()
	if all_upper(spl_line[1]) or spl_line[1] in ['SY1', 'SY2'] or is_role(spl_line[1]):
		return spl_line[1]
	else:
		return spl_line[1] + '.' + spl_line[2].replace('"','') #get sense without quotes


def get_num_concepts(clause_list, pos_tag):
	'''Get the number of concepts with a certain pos-tag'''
	num_concepts = 0
	for clause in clause_list:
		if not all_upper(clause[1]) and not is_role(clause[1]) and not clause[1] in ['SY1', 'SY2']:
			try:
				cur_pos_tag = clause[2].split('.')[0][-1]
				if cur_pos_tag == 's':
					cur_pos_tag = 'a' # a and a are both adjectives
				if cur_pos_tag == pos_tag:
					num_concepts += 1
			except:
				print('Strange concept clause', " ".join(clause))
	return num_concepts


def get_detailed_results(match):
	'''Get more detailed results regarding matching concepts, for nouns, verbs, adjectives and adverbs'''
	match_division = [0, 0, 0, 0, 0, 0, 0, 0]
	possible_tags = ['n','v','a','r'] 	# possible tags of WordNet concepts currently
	event_tags = ['v','a'] 				# tags that introduce events
	for clause in [x[0] for x in match]:
		spl_line = clause.split()
		if '%' in spl_line:
			spl_line = spl_line[0:spl_line.index('%')]
		if all_upper(spl_line[1]) or spl_line[1] in ['SY1', 'SY2']:
			match_division[0] += 1
		elif is_role(spl_line[1]):
			match_division[1] += 1
		else: #concepts, but also division of concepts
			match_division[2] += 1
			try:
				pos_tag = spl_line[2].split('.')[0][-1]
				if pos_tag == 's': #actually s and a are the same
					pos_tag = 'a'
				if pos_tag in possible_tags:
					match_division[possible_tags.index(pos_tag) + 3] += 1
				if pos_tag in event_tags:
					match_division[7] += 1
			except:
				print('Strange concept clause:', " ".join(spl_line))
	return match_division


def get_mappings(clause_pairs, prod_drs, gold_drs, prec, rec, significant):
	'''Output the matching and non-matching clauses'''

	# Flatten lists of lists to get list of indices
	indices_prod = prod_drs.op_two_vars_idx + prod_drs.op_two_vars_abs_idx1 + prod_drs.op_two_vars_abs_idx2 + prod_drs.op_three_vars_idx + prod_drs.roles_two_abs_idx + prod_drs.roles_abs_idx1 + prod_drs.roles_abs_idx2 + prod_drs.roles_idx + prod_drs.concepts_idx
	indices_gold = gold_drs.op_two_vars_idx + gold_drs.op_two_vars_abs_idx1 + gold_drs.op_two_vars_abs_idx2 + gold_drs.op_three_vars_idx + gold_drs.roles_two_abs_idx + gold_drs.roles_abs_idx1 + gold_drs.roles_abs_idx2 + gold_drs.roles_idx + gold_drs.concepts_idx

	idv_dict = {} #idv_dict is used to save information about individual matches, e.g. what is the F-score for "Theme" only
	match, no_match, orig_prod_idx_match, orig_gold_idx_match  = [], [], [], []

	# First get all matches and order them based on gold occurence
	ordered_clauses = [[indices_prod[idx2], indices_gold[idx1]] for idx1, idx2 in clause_pairs]
	for idx_prod, idx_gold in sorted(ordered_clauses, key=lambda x: x[1]):
		match.append([prod_drs.original_clauses[idx_prod], gold_drs.original_clauses[idx_gold]])
		orig_prod_idx_match.append(idx_prod)
		orig_gold_idx_match.append(idx_gold)

		#Save specific information about clause ID (Theme, work.n.01, IMP) here
		clause_id = get_clause_id(prod_drs.original_clauses[idx_prod])
		if clause_id not in idv_dict:
			idv_dict[clause_id] = [1, 1, 1] #contains [match_num, prod_num, gold_num]
		else:
			idv_dict[clause_id] = [x + y for x, y in zip(idv_dict[clause_id], [1, 1, 1])]

	# Get which indices match and do no match (do sorted for no match so that we have some kind of order still)
	no_match_prod = sorted([x for x in range(len(prod_drs.original_clauses)) if x not in orig_prod_idx_match])
	no_match_gold = sorted([x for x in range(len(gold_drs.original_clauses)) if x not in orig_gold_idx_match])

	# Populate the no match part of what we will print later (is a bit cumbersome because we want aligned printing later)
	for num in no_match_prod:
		no_match.append([prod_drs.original_clauses[num], ''])

		# Save individual information again here
		clause_id = get_clause_id(prod_drs.original_clauses[num])
		if clause_id not in idv_dict:
			idv_dict[clause_id] = [0, 1, 0] #contains [match_num, prod_num, gold_num]
		else:
			idv_dict[clause_id][1] += 1 #else only add 1 to prod_num

	for num in no_match_gold:
		found = False
		for no_match_item in no_match:
			if no_match_item[1] == '':  # there is still space to add
				no_match_item[1] = gold_drs.original_clauses[num]
				found = True
				break
		if not found:  # no space to add, create new row
			no_match.append(['', gold_drs.original_clauses[num]])

		# Save individual information again here
		clause_id = get_clause_id(gold_drs.original_clauses[num])
		if clause_id not in idv_dict:
			idv_dict[clause_id] = [0, 0, 1] #contains [match_num, prod_num, gold_num]
		else:
			idv_dict[clause_id][2] += 1 #else only add 1 to gold_num

	# Check if results make sense compared to our previous result
	print_prec = round(len(match) / float(prod_drs.total_clauses), significant)
	print_rec = round(len(match) / float(gold_drs.total_clauses), significant)
	if print_prec != prec or print_rec != rec:
		raise ValueError(
			"Error when calculating precision/recall for printing clauses:\nPrint prec {0} vs search prec {1}\nPrint rec {0} vs search rec {1}\n".format(print_prec, prec, print_rec, rec))

	# Print to screen and possibly to file later
	print_match = create_tab_list(match, '\n## Matching clauses ##\n', "| ")
	print_no_match = create_tab_list(no_match, '\n## Non-matching clauses ##\n', "| ")

	# Also get number of matching operator/roles/concepts for more detailed results
	match_division = get_detailed_results(match)
	return print_match, print_no_match, match_division, idv_dict


def get_matching_clauses(arg_list, incorrect_alignments_file = None, incorrect_alignments_file_fc = None, input_sequence = None):
	'''Function that gets matching clauses (easier to parallelize)'''
	start_time = time.time()
	# Unpack arguments to make things easier
	prod_t, gold_t, args, single, original_prod, original_gold, en_sense_dict, signature = arg_list
	# Create DRS objects
	prod_drs, gold_drs = DRS(signature), DRS(signature)
	prod_drs.prefix, gold_drs.prefix = 'a', 'b' # Prefixes are used to create standardized variable-names
	prod_drs.file_name, gold_drs.file_name = args.f1, args.f2
	prod_drs.original_clauses, gold_drs.original_clauses = original_prod, original_gold

	# Get the specific clauses here
	prod_drs.get_specific_clauses(prod_t, en_sense_dict, args)
	gold_drs.get_specific_clauses(gold_t, en_sense_dict, args)

	if single and (args.max_clauses > 0 and ((prod_drs.total_clauses > args.max_clauses) or (gold_drs.total_clauses > args.max_clauses))):
		print('Skip calculation of DRS, more clauses than max of {0}'.format(args.max_clauses))
		return 'skip'

	len_incorrect_alignments = 0
	if args.stats: #only do stats
		return [0, prod_drs.total_clauses, gold_drs.total_clauses, [], 0, 0, 0, len(prod_drs.var_map)]  # only care about clauses and var count, skip calculations
	else:
		# Do the hill-climbing for the matching here
		(best_mapping, best_match_num, found_idx, smart_fscores, clause_pairs, alignment) = get_best_match(prod_drs, gold_drs, args, single)
		(precision, recall, best_f_score) = compute_f(best_match_num, prod_drs.total_clauses, gold_drs.total_clauses, args.significant, False)

		if incorrect_alignments_file is not None and incorrect_alignments_file_fc is not None:
			incorrect_alignments = filter_matching_clauspairs_by_alignment(clause_pairs, alignment, ignore_casing = args.evaluate_token_ignore_casing, match_tokenIdxs = False,
                                                        match_roughly= args.match_roughly, return_incorrect= True, input_sequences = input_sequence)
			if len(incorrect_alignments) > 0:
				len_incorrect_alignments = len(incorrect_alignments)
				incorrect_alignments_file.write("\nSentence: " + input_sequence)
				incorrect_alignments_file.write("\n" +  "Prec, Rec, F-score: " + str(precision) + ", " + str(recall) + ", " + str(best_f_score))
				incorrect_alignments_file.write("\n\n")
				incorrect_alignments_file.write("\n".join([str(v) for k,v in incorrect_alignments.items()]))
				incorrect_alignments_file.write("\n\n")
				incorrect_alignments_file.write("Prod:\n" + "\n".join(original_prod))
				incorrect_alignments_file.write("\n\n")
				incorrect_alignments_file.write("Gold:\n" + "\n".join(original_gold))
				incorrect_alignments_file.write("\n")
				incorrect_alignments_file.write("##################################################################")
				incorrect_alignments_file.write("\n\n\n")

				if precision == 1 and recall == 1:
					incorrect_alignments_file_fc.write("\nSentence: " + input_sequence)
					incorrect_alignments_file_fc.write("\n" + "Prec, Rec, F-score: " + str(precision) + ", " + str(recall) + ", " + str(best_f_score))
					incorrect_alignments_file_fc.write("\n\n")
					incorrect_alignments_file_fc.write("\n".join([str(v) for k, v in incorrect_alignments.items()]))
					incorrect_alignments_file_fc.write("\n\n")
					incorrect_alignments_file_fc.write("Prod:\n" + "\n".join(original_prod))
					incorrect_alignments_file_fc.write("\n\n")
					incorrect_alignments_file_fc.write("Gold:\n" + "\n".join(original_gold))
					incorrect_alignments_file_fc.write("\n")
					incorrect_alignments_file_fc.write("##################################################################")
					incorrect_alignments_file_fc.write("\n\n\n")

		clause_pairs_correct_alignment_token, best_match_num_correct_alignment_token, clause_pairs_correct_alignment_idxs, best_match_num_correct_alignment_idxs, clause_pairs_correct_alignment_full, best_match_num_correct_alignment_full = 0,0,0,0,0,0
		if (args.reference_input_token and args.evaluate_token):
			clause_pairs_correct_alignment_token = filter_matching_clauspairs_by_alignment(clause_pairs, alignment, match_alignmentToken=True, ignore_casing = args.evaluate_token_ignore_casing, match_tokenIdxs = False,
                                                                                     match_roughly= args.match_roughly, input_sequences= input_sequence)
			best_match_num_correct_alignment_token = len(clause_pairs_correct_alignment_token)
		if args.reference_input_token and args.evaluate_indices:
			clause_pairs_correct_alignment_idxs = filter_matching_clauspairs_by_alignment(clause_pairs, alignment, match_alignmentToken=False, ignore_casing=args.evaluate_token_ignore_casing, match_tokenIdxs=True,
                                                                                          match_roughly= args.match_roughly, input_sequences= input_sequence)
			best_match_num_correct_alignment_idxs = len(clause_pairs_correct_alignment_idxs)
		if args.reference_input_token and args.evaluate_indices and args.evaluate_token:
			clause_pairs_correct_alignment_full = filter_matching_clauspairs_by_alignment(clause_pairs, alignment, match_alignmentToken=True, ignore_casing=args.evaluate_token_ignore_casing, match_tokenIdxs=True,
																							 match_roughly=args.match_roughly, input_sequences=input_sequence)
			best_match_num_correct_alignment_full = len(clause_pairs_correct_alignment_full)

		if not args.no_mapping:
			# Print clause mapping if -prin is used and we either do multiple scores, or just had a single DRS
			print_match, print_no_match, match_division, idv_dict = get_mappings(clause_pairs, prod_drs, gold_drs, precision, recall, args.significant)
			# Print to file or/and screen
			if args.prin and (args.ms or single) and not args.no_mapping:

				# Print rewrites of concepts that were performed (only for single instances)
				rewrite_list = create_tab_list(prod_drs.rewritten_concepts + gold_drs.rewritten_concepts, '\n## Normalized concepts to synset ID ##\n', '-> ')
				uniq_rewrite = []
				[uniq_rewrite.append(item) for item in rewrite_list if item not in uniq_rewrite] #only unique transformations, but keep order

				# Print match and non-match to screen
				for print_line in print_match: print(print_line)
				for print_line in print_no_match: print(print_line)
		else:
			match_division = []
			idv_dict = {}

		# For single DRS we print results later on anyway
		prod_clause_division = [prod_drs.num_operators, prod_drs.num_roles, prod_drs.num_concepts, get_num_concepts(prod_drs.concepts, 'n'), get_num_concepts(prod_drs.concepts, 'v'), get_num_concepts(prod_drs.concepts, 'a'), get_num_concepts(prod_drs.concepts, 'r'), get_num_concepts(prod_drs.concepts, 'v') + get_num_concepts(prod_drs.concepts, 'a')]
		gold_clause_division = [gold_drs.num_operators, gold_drs.num_roles, gold_drs.num_concepts, get_num_concepts(gold_drs.concepts, 'n'), get_num_concepts(gold_drs.concepts, 'v'), get_num_concepts(gold_drs.concepts, 'a'), get_num_concepts(gold_drs.concepts, 'r'), get_num_concepts(gold_drs.concepts, 'v') + get_num_concepts(gold_drs.concepts, 'a')]
		if args.ms and not single:
			print_results([[best_match_num, prod_drs.total_clauses, gold_drs.total_clauses, smart_fscores, found_idx, match_division, prod_clause_division, gold_clause_division, len(prod_drs.var_map), idv_dict]],
						  False, start_time, single, args)
		return [best_match_num, prod_drs.total_clauses, gold_drs.total_clauses, smart_fscores, found_idx, match_division, prod_clause_division, gold_clause_division, len(prod_drs.var_map), idv_dict, best_match_num_correct_alignment_token, best_match_num_correct_alignment_full, best_match_num_correct_alignment_idxs, len_incorrect_alignments]

def filter_matching_clauspairs_by_alignment(clause_pairs, alignment, ignore_casing = False, match_alignmentToken = True, match_tokenIdxs = True, match_roughly = False,
											return_incorrect = False, input_sequences = None):
	if set(clause_pairs.keys()) != set(alignment.keys()):
		raise ValueError("Alignment and clause-pair dictionaries have different key-sets!")

	res = {}
	for keypair in clause_pairs.keys():
		alignment_prod, alignment_gold = alignment[keypair]

		if match_alignmentToken and not match_tokenIdxs:
			match = alignment_matchTokens(alignment_prod[0], alignment_gold[0], ignore_casing = ignore_casing, match_roughly= match_roughly, input_sequences = input_sequences)
		elif not match_alignmentToken and match_tokenIdxs:
			match = alignment_prod[1] == alignment_gold[1] and alignment_prod[2] == alignment_gold[2]
		else:
			tokenMatch = alignment_matchTokens(alignment_prod[0], alignment_gold[0], ignore_casing=ignore_casing,
										  match_roughly=match_roughly, input_sequences=input_sequences)
			match = tokenMatch and alignment_prod[1] == alignment_gold[1] and alignment_prod[2] == alignment_gold[2]


		if return_incorrect:
			if not match:
				res[keypair] = alignment[keypair]
		else:
			if match:
				res[keypair] = clause_pairs[keypair]
	return res


def alignment_matchTokens(token_prod, token_gold, ignore_casing = False, match_roughly = False, input_sequences = None):
	if '~' in token_gold:
		if match_roughly:
			return rough_tokenMatch(token_prod.lower(), token_gold.lower(), check_using_inclusion = True, input_sequences = input_sequences) if ignore_casing \
				else rough_tokenMatch(token_prod, token_gold, check_using_inclusion = True, input_sequences = input_sequences)
		else:
			return token_prod.lower() in token_gold.lower() if ignore_casing else token_prod in token_gold
	else:
		if match_roughly:
			return rough_tokenMatch(token_prod.lower(), token_gold.lower(), input_sequences = input_sequences) if ignore_casing \
				else rough_tokenMatch(token_prod, token_gold, input_sequences = input_sequences)
		else:
			return token_prod.lower() == token_gold.lower() if ignore_casing else token_prod == token_gold

def rough_tokenMatch(token_prod, token_gold, check_using_inclusion = False, input_sequences = None):
	'''
	tokens1 = word_tokenize(token_prod)
	tokens2 = word_tokenize(token_gold)

	# Initialize stemmer and lemmatizer
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()

	# Stem and lemmatize words
	stemmed_word1 = [stemmer.stem(token) for token in tokens1]
	stemmed_word2 = [stemmer.stem(token) for token in tokens2]

	lemmatized_word1 = [lemmatizer.lemmatize(token) for token in tokens1]
	lemmatized_word2 = [lemmatizer.lemmatize(token) for token in tokens2]

	# Check if words are equal, stemmed, lemmatized, or synonyms
	if check_using_inclusion:
		if token_prod in token_gold:
			return True
		elif stemmed_word1 in stemmed_word2:
			return True
		elif lemmatized_word1 in lemmatized_word2:
			return True
	else:
		if token_prod == token_gold:
			return True
		elif stemmed_word1 == stemmed_word2:
			return True
		elif lemmatized_word1 == lemmatized_word2:
			return True
	synonyms1 = set()
	synonyms2 = set()
	for token in tokens1:
		synonyms1.update({syn.name() for syn in wordnet.synsets(token)})
	for token in tokens2:
		synonyms2.update({syn.name() for syn in wordnet.synsets(token)})
	if synonyms1.intersection(synonyms2):
		return True
	'''
	if token_prod == 'unk':
		return token_prod == token_gold
	best_match = find_best_match_editdistance(token_prod, input_sequences)
	return best_match == token_gold or (check_using_inclusion and best_match in token_gold)


def find_best_match_editdistance(token_prod, input_sequences):
	best_match = None
	min_distance = 99999
	if input_sequences is None or input_sequences == "":
		print(1)
	for tok in [x.lower() for x in input_sequences.split()]:
		distance = editdistance.eval(token_prod, tok)
		if distance < min_distance:
			min_distance = distance
			best_match = tok
	return best_match

def print_results(res_list, no_print, start_time, single, args, verbose = True):
	'''Print the final or inbetween scores -- res_list has format '''

	# Calculate average scores
	total_match_num = sum([x[0] for x in res_list if x])
	total_match_num_correct_alignment_token = sum([x[10] for x in res_list if x])
	total_match_num_correct_alignment_full = sum([x[11] for x in res_list if x])
	total_match_num_correct_alignment_idx = sum([x[12] for x in res_list if x])
	count_fully_correct_drs = sum(1 for x in res_list if x[0] == x[1] == x[2])
	indices_fully_correct_drs = [index for index, x in enumerate(res_list) if x[0] == x[1] == x[2]]
	total_match_num_correct_drs = sum(x[0] for x in res_list if x[0] == x[1] == x[2])
	total_match_num_correct_drs_correct_alignment_token = sum(x[10] for x in res_list if x[0] == x[1] == x[2])
	total_match_num_correct_drs_correct_alignment_full = sum(x[11] for x in res_list if x[0] == x[1] == x[2])
	total_match_num_correct_drs_correct_alignment_idx = sum(x[12] for x in res_list if x[0] == x[1] == x[2])

	num_clauses_gold = [x[2] for x in res_list if x[0] == x[1] == x[2]]
	num_clauses_gold_correct_drs = [x[2] for x in res_list if x[0] == x[1] == x[2]]


	total_test_num = sum([x[1] for x in res_list if x])
	total_gold_num = sum([x[2] for x in res_list if x])
	found_idx = round(float(sum([x[4] for x in res_list if x])) / float(len([x[4] for x in res_list if x])), args.significant)
	runtime = round(time.time() - start_time, args.significant)
	match_division = [x[5] for x in res_list]
	prod_division = [x[6] for x in res_list]
	gold_division = [x[7] for x in res_list]
	count_alignment_errors_token_only = total_match_num - total_match_num_correct_alignment_token
	count_alignment_errors_token_and_indices = total_match_num - total_match_num_correct_alignment_full
	count_alignment_errors_indices_only = total_match_num - total_match_num_correct_alignment_idx

	#print(indices_fully_correct_drs)

	# Calculate detailed F-scores for clauses, roles, concepts etc
	name_list = ['operators','roles','concepts','nouns','verbs','adjectives','adverbs','events']
	res_dict = {}
	for idx, name in enumerate(name_list):
		res_dict[name] = compute_f(sum([x[idx] for x in match_division]), sum([x[idx] for x in prod_division]), sum([x[idx] for x in gold_division]), args.significant, False)

	# Output document-level score (a single f-score for all DRS pairs in two files)
	(precision, recall, best_f_score) = compute_f(total_match_num, total_test_num, total_gold_num, args.significant, False)

	alignment_accuracy_token, alignment_accuracy_idx, alignment_accuracy_full, alignment_accuracy_fully_correct_token, alignment_accuracy_fully_correct_idx, alignment_accuracy_fully_correct_full = 0,0,0,0,0,0
	precision_align_token, recall_align_token, best_f_score_align_token, precision_align_idx, recall_align_idx, best_f_score_align_idx, precision_align_full, recall_align_full, best_f_score_align_full = 0,0,0,0,0,0,0,0,0
	if (args.reference_input_token):
		(precision_align_token, recall_align_token, best_f_score_align_token) = compute_f(total_match_num_correct_alignment_token, total_test_num, total_gold_num, args.significant, False)
		(precision_align_idx, recall_align_idx, best_f_score_align_idx) = compute_f(total_match_num_correct_alignment_idx, total_test_num, total_gold_num, args.significant, False)
		(precision_align_full, recall_align_full, best_f_score_align_full) = compute_f(total_match_num_correct_alignment_full, total_test_num, total_gold_num, args.significant, False)

		alignment_accuracy_token = total_match_num_correct_alignment_token / total_match_num
		alignment_accuracy_idx = total_match_num_correct_alignment_idx / total_match_num
		alignment_accuracy_full = total_match_num_correct_alignment_full / total_match_num

		alignment_accuracy_fully_correct_token = total_match_num_correct_drs_correct_alignment_token / total_match_num_correct_drs
		alignment_accuracy_fully_correct_idx = total_match_num_correct_drs_correct_alignment_idx / total_match_num_correct_drs
		alignment_accuracy_fully_correct_full = total_match_num_correct_drs_correct_alignment_full / total_match_num_correct_drs

	if not res_list:
		return []  # no results for some reason
	elif no_print:  # averaging over multiple runs, don't print results
		return [precision, recall, best_f_score]
	else:
		if verbose:
			print('\n## Clause information ##\n')
			print('Clauses prod : {0}'.format(total_test_num))
			print('Clauses gold : {0}\n'.format(total_gold_num))
			if args.max_clauses > 0:
				print('Max number of clauses per DRS:  {0}\n'.format(args.max_clauses))
			print('## Main Results ##\n')
			if not single:
				print('All shown number are micro-averages calculated over {0} DRS-pairs\n'.format(len(res_list)))
			print('Matching clauses: {0}'.format(total_match_num))
			print('Fully correct DRS: {0} ({1} out of {2})\n'.format(round(count_fully_correct_drs / len(res_list), 2), count_fully_correct_drs, len(res_list)))
			print("Precision: {0}".format(round(precision, args.significant)))
			print("Recall   : {0}".format(round(recall, args.significant)))
			print("F-score  : {0}".format(round(best_f_score, args.significant)))
			if (args.reference_input_token and args.evaluate_token):
				print("-------- Alignment ---------")
				#print("Alignment Accuracy (without indices): {0}".format(round(alignment_accuracy, args.significant)))
				#print("Alignment Accuracy for fully correct DRS (without indices): {0}".format(round(alignment_accuracy_fully_correct, args.significant)))
				#print("Alignment Accuracy (with indices): {0}".format(round(alignment_accuracy_idx, args.significant)))
				#print("Alignment Accuracy for fully correct DRS (with indices): {0}".format(round(alignment_accuracy_fully_correct_idx, args.significant)))
				#print("Count Alignment Errors (without indices): {0}".format(count_alignment_errors_token_only))
				#print("Count Alignment Errors (with indices): {0}".format(count_alignment_errors_token_and_indices))
				#print("F-score  : {0}".format(round(best_f_score_align, args.significant)))

		# Print specific output here
		if args.prin:
			print('\n## Detailed precision, recall, F-score ##\n')
			for idx in range(0,len(name_list)):
				print('Prec, rec, F1 {0}: {1}, {2}, {3}'.format(name_list[idx], res_dict[name_list[idx]][0], res_dict[name_list[idx]][1], res_dict[name_list[idx]][2]))
			if args.smart == 'conc':
				smart_conc = compute_f(sum([y[0] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, args.significant, True)
				print('Smart F-score concepts: {0}\n'.format(smart_conc))

			# For a single DRS we can print some more information
			if single:
				print('\n## Restarts and processing time ##\n')
				print('Num restarts specified       : {0}'.format(args.restarts))
				print('Found best mapping at restart: {0}'.format(int(found_idx)))

		# Print detailed results in an html file for the CodaLab usage
		if args.codalab:
			#global ill_drs_ids # number of ill DRSs found in the system output
			counts = (total_test_num, total_gold_num, total_match_num)
			measures = (precision, recall, best_f_score)
			html_content = coda_html(len(res_list), ill_drs_ids, counts, measures, name_list, res_dict)
			with codecs.open(args.codalab+'.html', 'w', encoding='UTF-8') as html:
				html.write(html_content)

	if verbose:
		print('Total processing time: {0} sec'.format(runtime))
	return [best_f_score, best_f_score_align_token, best_f_score_align_full, best_f_score_align_idx, alignment_accuracy_token, alignment_accuracy_fully_correct_token, alignment_accuracy_full, alignment_accuracy_fully_correct_full, alignment_accuracy_idx, alignment_accuracy_fully_correct_idx, count_alignment_errors_token_only, count_alignment_errors_token_and_indices, count_alignment_errors_indices_only, count_fully_correct_drs, len(res_list), num_clauses_gold, num_clauses_gold_correct_drs]


def check_input(clauses_prod_list, original_prod, original_gold, clauses_gold_list, baseline, f1, max_clauses, single, verbose = False):
	'''Check if the input is valid -- or fill baseline if that is asked'''
	if baseline:  # if we try a baseline DRS, we have to fill a list of this baseline
		if len(clauses_prod_list) == 1:
			if verbose:
				print('Testing baseline DRS vs {0} DRSs...\n'.format(len(clauses_gold_list)))
			clauses_prod_list = fill_baseline_list(clauses_prod_list[0], clauses_gold_list)
			original_prod = fill_baseline_list(original_prod[0], original_gold)
		else:
			raise ValueError("Using --baseline, but there is more than 1 DRS in prod file")
	elif len(clauses_prod_list) != len(clauses_gold_list):
		if verbose:
			print("Number of DRSs not equal, {0} vs {1}, exiting...".format(len(clauses_prod_list), len(clauses_gold_list)))
		sys.exit(0)
	elif len(clauses_prod_list) == 0 and len(clauses_gold_list) == 0:
		if verbose:
			print("Both DRSs empty, exiting...")
		sys.exit(0)
	elif not single:
		if verbose:
			print('Comparing {0} DRSs...\n'.format(len(clauses_gold_list)))

	# Print number of DRSs we skip due to the -max_clauses parameter
	if max_clauses > 0 and not single:
		if verbose:
			print('Skipping {0} DRSs due to their length exceeding {1} (--max_clauses)\n'.format(len([x for x, y in zip(clauses_gold_list, clauses_prod_list) if len(x) > max_clauses or len(y) > max_clauses]), max_clauses))
	return original_prod, clauses_prod_list


class DRS:
	'''Main DRS class with methods to process and rewrite the clauses'''

	def __init__(self, signature):
		# List of operators who's variables are also boxes, e.g. b0 NOT b1
		self.op_boxes = [key for key in signature if signature[key][1] in ['bb', 'bbb']]
		# List of operators for which b OP y1 y2 == b1 OP y2 y1
		self.inv_boxes = ['APX', 'NEQ', 'EQU', 'TAB']
		# List for different types of clauses
		self.op_two_vars, self.op_two_vars_idx = [], []
		self.op_two_vars_abs1, self.op_two_vars_abs_idx1 = [], []
		self.op_two_vars_abs2, self.op_two_vars_abs_idx2 = [], []
		self.op_three_vars, self.op_three_vars_idx = [], []
		self.roles_two_abs, self.roles_two_abs_idx = [], []
		self.roles_abs1, self.roles_abs_idx1 = [], []
		self.roles_abs2, self.roles_abs_idx2 = [], []
		self.roles, self.roles_idx = [], []
		self.concepts, self.concepts_idx = [], []

		# Lists and dicts to keep track of the variable information and rewritten concepts
		self.vars_seen = []
		self.type_vars = {}
		self.var_map = {}
		self.rewritten_concepts = []
		self.added_var_map = False
		self.token_refference = []

	def rename_var(self, var, var_type, args):
		'''Function that renames the variables in a standardized way'''
		if var in self.vars_seen:
			new_var = self.prefix + str(self.vars_seen.index(var))
		else:
			new_var = self.prefix + str(len(self.vars_seen))
			self.type_vars[new_var] = var_type
			self.vars_seen.append(var)
		self.var_map[new_var] = var
		return new_var


	def add_if_not_exists(self, list1, list2, to_add, to_add2):
		'''Add something to two list of lists if it is not there yet in the first one'''
		if to_add not in list1:
			list1.append(to_add)
			list2.append(to_add2)


	def add_concept_clauses(self, cur_clause, val0, idx, en_sense_dict, args):
		'''Add concepts clauses to our list of clauses'''
		# First rename the variable
		val3 = self.rename_var(cur_clause[3], 'x', args)
		# Then rewrite the concept to its synset ID in WordNet
		new_concept, new_sense, rewritten_values = rewrite_concept(cur_clause[1], cur_clause[2], en_sense_dict)
		# Maybe change concept/sense to a default
		if args.default_concept:
			new_concept = 'work'
			new_sense = '"n.01"'
		elif args.default_sense:
			new_sense = '"n.01"'
		if args.reference_input_token:
			self.add_if_not_exists(self.concepts, self.concepts_idx, (val0, new_concept, new_sense, val3,
																	  cur_clause[5], cur_clause[6], cur_clause[7]), idx)
		else:
			self.add_if_not_exists(self.concepts, self.concepts_idx, (val0, new_concept, new_sense, val3), idx)
		# Save rewritten concepts
		if rewritten_values:
			self.rewritten_concepts.append(rewritten_values)


	def add_role_clauses(self, cur_clause, val0, idx, args):
		'''Add clauses that have a Role as second item to our list of clauses'''
		# Check if we want to rewrite the role to a default
		role = 'Role' if args.default_role else cur_clause[1]

		# If second and third item are between quotes it belongs in roles_two_abs (b0 PartOf "speaker" "hearer")
		if between_quotes(cur_clause[2]) and between_quotes(cur_clause[3]):
			to_add = (val0, role, cur_clause[2], cur_clause[3]) if not args.reference_input_token else \
				(val0, role, cur_clause[2], cur_clause[3], cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.roles_two_abs, self.roles_two_abs_idx, to_add, idx)
		# If third item is between quotes it belongs in roles_abs1 (b0 PartOf "speaker" x2)
		elif between_quotes(cur_clause[2]):
			val2 = self.rename_var(cur_clause[3], 'x', args)
			to_add = (val0, role, cur_clause[2], val2) if not args.reference_input_token else \
				(val0, role, cur_clause[2], val2, cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.roles_abs1, self.roles_abs_idx1, to_add, idx)
		# If last item is between quotes it belongs in roles_abs2 (b0 PartOf x2 "speaker")
		elif between_quotes(cur_clause[3]):
			val2 = self.rename_var(cur_clause[2], 'x', args)
			to_add = (val0, role, val2, cur_clause[3]) if not args.reference_input_token else \
				(val0, role, val2, cur_clause[3], cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.roles_abs2, self.roles_abs_idx2, to_add, idx)
		# Otherwise it belongs in roles (b1 Patient x1 x2)
		else:
			val2 = self.rename_var(cur_clause[2], 'x', args)
			val3 = self.rename_var(cur_clause[3], 'x', args)
			to_add = (val0, role, val2, val3) if not args.reference_input_token else \
				(val0, role, val2, val3, cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.roles, self.roles_idx, to_add, idx)


	def add_operator_clauses(self, val0, cur_clause, idx, args):
		'''Add clauses that have an operator as second item to our list of close'''
		# Get var type of second variable
		var_type = 'b' if cur_clause[1] in self.op_boxes and cur_clause[1] != 'PRP' else 'x'

		# If second item between quotes it belongs in op_two_vars_abs1
		if between_quotes(cur_clause[2]):
			val2 = self.rename_var(cur_clause[3], var_type, args) #get renamed variable
			to_add = (val0, cur_clause[1], cur_clause[2], val2) if not args.reference_input_token else \
				(val0, cur_clause[1], cur_clause[2], val2, cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.op_two_vars_abs1, self.op_two_vars_abs_idx1, to_add, idx)
		# If last item is between quotes it belongs in op_two_vars_abs2 (b2 EQU t1 "now")
		elif between_quotes(cur_clause[3]):
			val2 = self.rename_var(cur_clause[2], var_type, args) #get renamed variable
			to_add = (val0, cur_clause[1], val2, cur_clause[3]) if not args.reference_input_token else \
				(val0, cur_clause[1], val2, cur_clause[3], cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.op_two_vars_abs2, self.op_two_vars_abs_idx2, to_add, idx)
		# Else it belongs in op_three_vars (b1 LES x1 x2)
		else:
			val2 = self.rename_var(cur_clause[2], var_type, args) #get renamed variable
			var_type = 'b' if cur_clause[1] in self.op_boxes else 'x'
			val3 = self.rename_var(cur_clause[3], var_type, args)
			to_add = (val0, cur_clause[1], val2, val3) if not args.reference_input_token else \
				(val0, cur_clause[1], val2, val3, cur_clause[5], cur_clause[6], cur_clause[7])
			self.add_if_not_exists(self.op_three_vars, self.op_three_vars_idx, to_add, idx)


	def get_specific_clauses(self, clause_list, en_sense_dict, args):
		'''Function that gets the specific clauses
		   Also renames them and changes their format to DRS format
		   Keep track of indices so we can easily print matching DRSs later
		   Different types:
				op_two_vars      : b0 REF x1
				op_two_vars_abs1 : b2 EQU t1 "now"
				op_two_vars_abs2 : b2 EQU "now" t1
				op_three_vars    : b1 LES x1 x2
				roles_abs1       : b0 PartOf x2 "speaker"
				roles_abs2       : b0 PartOf "speaker" x2
				roles_two_abs    : b0 PartOf "hearer" "speaker"
				roles            : b0 Patient x1 x2
				concepts         : b1 work "v.01" x2'''

		# Operators that have two box variables
		for idx, cur_clause in enumerate(clause_list):
			# Clause has three items and belongs in op_two_vars (b0 REF x1 ,  b0 NOT b1)

			if len(cur_clause) == 3 or (args.reference_input_token and len(cur_clause) == 7):
				val0 = self.rename_var(cur_clause[0], 'b', args)
				var_type = 'b' if cur_clause[1] in self.op_boxes else 'x'
				val2 = self.rename_var(cur_clause[2], var_type, args)
				if args.reference_input_token:
					origin_token, charseq_start, charseq_end = cur_clause[4], cur_clause[5], cur_clause[6]
					self.add_if_not_exists(self.op_two_vars, self.op_two_vars_idx, (val0, cur_clause[1], val2, origin_token, charseq_start, charseq_end), idx)
				else:
					self.add_if_not_exists(self.op_two_vars, self.op_two_vars_idx, (val0, cur_clause[1], val2), idx)
			# Clause has 4 items
			else:
				# First item always is a box variable
				val0 = self.rename_var(cur_clause[0], 'b', args)
				if all_upper(cur_clause[1]) or cur_clause[1] in ['SY1', 'SY2']: # Second item is an operator
					self.add_operator_clauses(val0, cur_clause, idx, args)
				elif is_role(cur_clause[1]): # Second item is a role
					self.add_role_clauses(cur_clause, val0, idx, args)
				else:                       # Otherwise it must be a concept (b1 work "v.01" x2)
					self.add_concept_clauses(cur_clause, val0, idx, en_sense_dict, args)

		# Get number of operator/role/concept clauses
		self.num_operators = len(self.op_two_vars) + len(self.op_two_vars_abs1) + len(self.op_two_vars_abs2) + len(self.op_three_vars)
		self.num_roles = len(self.roles) + len(self.roles_abs1) + len(self.roles_abs2) + len(self.roles_two_abs)
		self.num_concepts = len(self.concepts)
		self.total_clauses = self.num_operators + self.num_roles + self.num_concepts


def save_detailed_stats(all_dicts, args):
	'''Print detailed statistics to the screen, if args.detailed_stats > 0'''
	final_dict, f_dict = merge_dicts(all_dicts, args) #first merge all dictionaries in a single dict and also create dict with F-scores
	print_list = [[], [], []]
	print_headers = ['Operators', 'Roles', 'Concepts']
	print_line = 'Individual clause scores for'

	# Create lists to print when looping over the sorted dictionary
	for w in sorted(f_dict, key=f_dict.get, reverse=True):
		if final_dict[w][1] >= args.detailed_stats or final_dict[w][2] >= args.detailed_stats: #only print if it has a minimum occurrence in prod or gold
			if all_upper(w) or w in ['SY1', 'SY2']:
				print_list[0].append([w, str(f_dict[w]), str(final_dict[w][1]), str(final_dict[w][2]), str(final_dict[w][0])])
			elif is_role(w):
				print_list[1].append([w, str(f_dict[w]), str(final_dict[w][1]), str(final_dict[w][2]), str(final_dict[w][0])])
			else:
				print_list[2].append([w, str(f_dict[w]), str(final_dict[w][1]), str(final_dict[w][2]), str(final_dict[w][0])])

	# Now print a nicely aligned tab-list for the three types of clauses so it is easier to keep them straight
	for idx, item in enumerate(print_list):
		print('\n{0} {1}:\n'.format(print_line, print_headers[idx].lower()))
		to_print = create_tab_list([[print_headers[idx], 'F-score', 'Prod inst', 'Gold inst', "Match"]] + print_list[idx], [], '\t')
		for t in to_print: print(t)


def save_stats(all_clauses, all_vars, stat_file):
	'''Print and save statistics related to number of variables and clauses per DRSs'''
	max_num = 0
	for idx, num in enumerate(all_clauses):
		if num > max_num:
			max_num = num
			best_idx = idx

	drs_max_clauses, mean_clauses, mean_variables = best_idx + 1, float(sum(all_clauses)) / float(len(all_clauses)), float(sum(all_vars)) / float(len(all_vars))
	print('\nStatistics:\n\nDRS {0} has max clauses\nLen DRSs: {1}\nMean clauses: {2}\nMedian clauses: {3}\nMax clauses: {4}\n'.format(drs_max_clauses, len(all_clauses), mean_clauses, median(all_clauses), max(all_clauses)))
	print('Mean variables: {0}\nMedian variables: {1}\nMax variables: {2}\n'.format(mean_variables, median(all_vars), max(all_vars)))

	dump_lists = [all_clauses, all_vars]
	pickle.dump(dump_lists, open(stat_file, 'w'))


def main(args, verbose = True):
	'''Main function of counter score calculation'''
	start = time.time()

	# Read in English sense dict for rewriting concepts to synset ID
	# This is read from a Python import to speed things up (twice as fast as loading JSON every time)
	from wordnet_dict_en import en_sense_dict
	signature = get_signature(args.sig_file) # Get signature
	res = []

	# Get all the clauses and check if they are valid
	clauses_gold_list, original_gold, input_sequences = get_clauses(args.f2, signature, args.ill, args.reference_input_token, include_REF = args.include_ref, read_nl_sentences = True )

	clauses_prod_list, original_prod, _ = get_clauses(args.f1, signature, args.ill, args.reference_input_token, include_REF = args.include_ref)


	if args.alignment_errors:
		if os.path.exists(args.alignment_errors):
			# If the file exists, just open it
			alignerror_file = open( args.alignment_errors, 'a')
			alignerror_file_fc = open( args.alignment_errors + ".fc", 'a')
		else:
			# If the file doesn't exist, create it and then open it
			alignerror_file = open( args.alignment_errors, 'w+')
			alignerror_file_fc = open( args.alignment_errors + ".fc", 'w+')
	else:
		alignerror_file = None
		alignerror_file_fc = None

	# Count ill-DRSs in the system output
	global ill_drs_ids
	if args.codalab and args.ill == 'dummy':
		ill_drs_ids = [ i for (i, x) in enumerate(clauses_prod_list, start=1) if len(x) < 3
						and	next( (cl for cl in x if len(cl) > 1 and cl[1].startswith('alwayswrong')), False ) ]

	# Don't print the results each time if we do multiple runs
	no_print = True if args.runs > 1 else False
	single = True if len(clauses_gold_list) == 1 else False  # true if we are doing a single DRS

	# Check if correct input (number of instances, baseline, etc)
	original_prod, clauses_prod_list = check_input(clauses_prod_list, original_prod, original_gold, clauses_gold_list, args.baseline, args.f1, args.max_clauses, single, verbose = verbose)

	# Processing clauses
	for _ in range(args.runs):  # for experiments we want to more runs so we can average later
		arg_list = []
		for count, (prod_t, gold_t) in enumerate(zip(clauses_prod_list, clauses_gold_list)):
			arg_list.append([prod_t, gold_t, args, single, original_prod[count], original_gold[count], en_sense_dict, signature])

		# Parallel processing here
		if args.parallel == 1:  # no need for parallelization for p=1
			all_results = []
			for num_count, arguments in enumerate(arg_list):
				all_results.append(get_matching_clauses(arguments, alignerror_file, alignerror_file_fc, input_sequences[num_count]))
		else:
			all_results = multiprocessing.Pool(args.parallel).map(get_matching_clauses, arg_list, alignerror_file, alignerror_file_fc, input_sequences)

		# If we find results, print them in a nice way
		if all_results == ['skip']: #skip result
			pass
		elif all_results and all_results[0]:
			all_clauses = [x[1] for x in all_results]
			all_vars = [x[7] for x in all_results]
			if not args.stats:
				res.append(print_results(all_results, no_print, start, single, args, verbose))
		else:
			raise ValueError('No results found')

	# If multiple runs, print averages
	if res and args.runs > 1 and not args.stats and verbose:
		print('Average scores over {0} runs:\n'.format(args.runs))
		print('Precision: {0}'.format(round(float(sum([x[0] for x in res])) / float(args.runs), args.significant)))
		print('Recall   : {0}'.format(round(float(sum([x[1] for x in res])) / float(args.runs), args.significant)))
		print('F-score  : {0}'.format(round(float(sum([x[2] for x in res])) / float(args.runs), args.significant)))

	# print scores in scores.txt file for codalab usage
	if len(res) == 1 and args.runs == 1 and args.codalab:
		with codecs.open(args.codalab+'.txt', 'w', encoding='UTF-8') as scores_file:
			[[score_p, score_r, score_f]] = res
			scores_file.write("Precision: {}\nRecall: {}\nF-score: {}".format(*res[0]))

	# Sometimes we are also interested in (saving and printing) some statistics, do that here
	if args.stats and args.runs <= 1:
		save_stats(all_clauses, all_vars, args.stats)

	# We might also want to save all individual F-scores, usually for the sake of doing statistics. We print them to a file here
	if args.ms_file:
		with open(args.ms_file, 'w') as out_f:
			for items in all_results:
				_, _, f_score = compute_f(items[0], items[1], items[2], args.significant, False)
				if args.all_idv:
					print_line = " ".join([str(x) for x in [items[0], items[1], items[2]]])
					out_f.write(print_line + '\n')
				else:
					out_f.write(str(f_score) + '\n')
		out_f.close()

	# We might want to output statistics about individual types of clauses
	if args.detailed_stats > 0:
		save_detailed_stats([x[9] for x in all_results], args)

	#return [best_f_score, best_f_score_align, best_f_score_align_idx, alignment_accuracy,
	#				alignment_accuracy_fully_correct, alignment_accuracy_idx, alignment_accuracy_fully_correct_idx]

	return (round(float(sum([x[0] for x in res])) / float(args.runs), args.significant), round(float(sum([x[1] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[2] for x in res])) / float(args.runs), args.significant), round(float(sum([x[3] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[4] for x in res])) / float(args.runs), args.significant), round(float(sum([x[5] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[6] for x in res])) / float(args.runs), args.significant), round(float(sum([x[7] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[8] for x in res])) / float(args.runs), args.significant), round(float(sum([x[9] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[10] for x in res])) / float(args.runs), args.significant), round(float(sum([x[11] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[12] for x in res])) / float(args.runs), args.significant), round(float(sum([x[13] for x in res])) / float(args.runs), args.significant),
			round(float(sum([x[14] for x in res])) / float(args.runs), args.significant))

ERROR_LOG = sys.stderr
DEBUG_LOG = sys.stderr

if __name__ == "__main__":
	args = build_arg_parser()
	main(args)
