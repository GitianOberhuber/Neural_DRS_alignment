#!/bin/bash
set -eu -o pipefail

# Give some general information
echo "Test if preprocessing and training AllenNLP models works as expected"
echo "The training/testing of the models can take quite some time and needs to run on GPU"
echo "Expect this script to run well over an hour"; sleep 5
echo
echo "Note: we expect that you followed the setup instructions in the general README"
echo "In other words, the data/ and DRS_parsing/ folders exist and have the correct content"
echo ; sleep 5


##### Preprocessing ######

# First test the preprocessing calls in the README
echo "First do all the English preprocessing as specified in AllenNLP.md" ;sleep 3

for type in dev test; do
	python src/preprocess.py --input_file data/3.0.0/en/gold/${type}.txt -v rel -r word -cd .tgt --drss_only
done

for type in dev test; do
	python src/preprocess.py --sentence_file data/3.0.0/en/gold/dev.txt.raw -r char -cs .char.sent --sents_only -c feature
done

for type in dev test; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold/${type}.txt.raw.tok.gold  3< data/3.0.0/en/gold/${type}.txt.tgt > data/3.0.0/en/gold/${type}.alp
done

for type in train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold_silver/${type}.txt -v rel -r word -cd .tgt --drss_only
done

for type in train; do
	python src/preprocess.py --sentence_file data/3.0.0/en/gold_silver/${type}.txt.raw -r char -cs .char.sent --sents_only -c feature
done

for type in train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold_silver/${type}.txt.raw.tok  3< data/3.0.0/en/gold_silver/${type}.txt.tgt > data/3.0.0/en/gold_silver/${type}.alp
done




##### EXPERIMENTS ######

echo "Now do experiments for English/German, mostly with BERT models" ; sleep 5

# Script that tests our setup for AllenNLP: test small models of our most important model settings
PIPELINE="src/allennlp_scripts/pipeline.sh"
CONFIG="config/allennlp/en_goldsilv/"
EXPS="experiments/allennlp/en_goldsilv/"

mkdir -p $EXPS


# Test BERT model
$PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ normal en


echo
echo "If you see this, the experiments did not throw any errors"
