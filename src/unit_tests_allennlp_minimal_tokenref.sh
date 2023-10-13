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

for type in train dev test; do
	python src/preprocess.py --input_file DRS_parsing/data/pmb-3.0.0/gold/${type}.txt -v rel -r word -cd .tgt -rt --drss_only
done

for type in train dev test; do
	python src/preprocess.py --sentence_file DRS_parsing/data/pmb-3.0.0/gold/dev.txt.raw -r char -cs .char.sent --sents_only -c feature
done

for type in train dev test; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < DRS_parsing/data/pmb-3.0.0/gold/${type}.txt.raw.tok  3< DRS_parsing/data/pmb-3.0.0/gold/${type}.txt.tgt > DRS_parsing/data/pmb-3.0.0/gold/${type}.alp
done


# Download and unzip PMB GloVe embeddings if the file does not exist yet
if [[ -f "emb/glove.840B.300d.pmb.txt" ]]; then
	echo "Glove embeddings already downloaded, skip"
else
	echo "Downloading and unpacking PMB GloVe embeddings, see emb/"
	mkdir -p emb; cd emb
	wget "http://www.let.rug.nl/rikvannoord/embeddings/glove_pmb.zip"
	unzip glove_pmb.zip; rm glove_pmb.zip;  cd ../
fi

##### EXPERIMENTS ######

echo "Now do experiments for English/German, mostly with BERT models" ; sleep 5

# Script that tests our setup for AllenNLP: test small models of our most important model settings
PIPELINE="src/allennlp_scripts/pipeline_rt.sh"
CONFIG="config/allennlp/en/"
EXPS="experiments/allennlp/en/"

mkdir -p $EXPS

# Remove all current models in exps to have a fresh new test
rm -r $EXPS/* || true

# Experiments are in order of likelihood to fail


# Test BERT model
$PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ normal en

echo
echo "If you see this, the experiments did not throw any errors"
