#!/bin/bash
set -eu -o pipefail

option_al=false
for arg in "$@"; do
    if [ "$arg" == "-al" ]; then
        option_al=true
        break  # Exit the loop since we found the option
    fi
done


##### Preprocessing ######
if $option_al; then
    echo "Performing preprocessing WITH alignment..." ;sleep 1
    echo "Creating gold .alp file..." ;sleep 1
    for type in dev test train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold/${type}.txt -v rel -r word -cd .tgt --drss_only -al
    done
    for type in dev test train; do
	python src/tokenize_raw_tok.py --input_file data/3.0.0/en/gold/${type}.txt.raw
    done

    for type in dev test train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold/${type}.txt.raw.tok  3< data/3.0.0/en/gold/${type}.txt.tgt.al > data/3.0.0/en/gold/${type}.al.alp
    done
    
    PIPELINE="src/allennlp_scripts/pipeline_al.sh"
    CONFIG="config/allennlp/en_default/en_gold_tok/"
    EXPS="experiments/allennlp/en_default/en_gold_tok/"
    RES="experiments/allennlp/en_default/en_gold_tok/bert/run1"
else
    echo "Performing preprocessing WITHOUT alignment..." ;sleep 1
    echo "Creating gold .alp file..." ;sleep 1
    for type in dev test train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold/${type}.txt -v rel -r word -cd .tgt --drss_only
    done
    for type in dev test train; do
	python src/tokenize_raw_tok.py --input_file data/3.0.0/en/gold/${type}.txt.raw
    done

    for type in dev test train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold/${type}.txt.raw.tok  3< data/3.0.0/en/gold/${type}.txt.tgt > data/3.0.0/en/gold/${type}.alp
    done
    
    PIPELINE="src/allennlp_scripts/pipeline.sh"
    CONFIG="config/allennlp/en_default/en_gold_nontok/"
    EXPS="experiments/allennlp/en_default/en_gold_nontok/"
    RES="experiments/allennlp/en_default/en_gold_nontok/bert/run1"
fi
    
echo "Training model on gold data..." ;sleep 1

mkdir -p $EXPS
$PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ normal en

EPOCHS=$(cat "${CONFIG}/bert.json" | sed -e 's/,/\n/g' | grep 'num_epochs"' | grep -oP "\d+")
python DRS_parsing/evaluation/summarize_results.py -d1 $RES -d2 data/3.0.0/en/gold -si DRS_parsing/evaluation/clf_signature.yaml -e $EPOCHS > "${RES}/result_overview.txt"

echo
echo "If you see this, the experiments did not throw any errors"
