#!/bin/bash
set -eu -o pipefail

option_rt=false
for arg in "$@"; do
    if [ "$arg" == "-rt" ]; then
        option_rt=true
        break  # Exit the loop since we found the option
    fi
done


##### Preprocessing ######
if $option_rt; then
    echo "Performing preprocessing WITH token references..." ;sleep 1
    echo "Creating gold .alp file..." ;sleep 1
    for type in dev test train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold/${type}.txt -v rel -r word -cd .tgt --drss_only -rt
    done
    for type in dev test train; do
	python src/tokenize_raw_tok.py --input_file data/3.0.0/en/gold/${type}.txt.raw
    done

    for type in dev test train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold/${type}.txt.raw.tok  3< data/3.0.0/en/gold/${type}.txt.tgt.rt > data/3.0.0/en/gold/${type}.rt.alp
    done
    
    #gold_silver contains only training data, dev and test are found in gold
    #both training data of gold_silver and only gold are needed , for training and finetuning respectively
    echo "Creating gold+silver .alp file..." ;sleep 1
    for type in train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold_silver/${type}.txt -v rel -r word -cd .tgt --drss_only -rt
    done

    for type in train; do
	python src/tokenize_raw_tok.py --input_file data/3.0.0/en/gold_silver/${type}.txt
    done

    for type in train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold_silver/${type}.txt.raw.tok.rt  3< data/3.0.0/en/gold_silver/${type}.txt.tgt > data/3.0.0/en/gold_silver/${type}.rt.alp
    done
    
    PIPELINE="src/allennlp_scripts/pipeline_rt.sh"
    CONFIG="config/allennlp/en_gold_tok/"
    EXPS="experiments/allennlp/en_gold_tok/"
    
    echo "Training model on gold data..." ;sleep 1

    mkdir -p $EXPS
    $PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ normal en
    echo "Fine-tuning model on gold data..." ;sleep 1
    CONFIG="config/allennlp/en_goldsilv_tok_tune/" #only difference from old config file: train_data_path points to gold data (as opposed to gold+silver)
else
    echo "Performing preprocessing WITHOUT token references..." ;sleep 1
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
    
    #gold_silver contains only training data, dev and test are found in gold
    #both training data of gold_silver and only gold are needed , for training and finetuning respectively
    echo "Creating gold+silver .alp file..." ;sleep 1
    for type in train; do
	python src/preprocess.py --input_file data/3.0.0/en/gold_silver/${type}.txt -v rel -r word -cd .tgt --drss_only
    done

    for type in train; do
	python src/tokenize_raw_tok.py --input_file data/3.0.0/en/gold_silver/${type}.txt
    done

    for type in train; do
	while IFS= read -r line1 && IFS= read -r line2 <&3; do
		echo -e "${line1}\t${line2}"
	done < data/3.0.0/en/gold_silver/${type}.txt.raw.tok  3< data/3.0.0/en/gold_silver/${type}.txt.tgt > data/3.0.0/en/gold_silver/${type}.alp
    done
    
    PIPELINE="src/allennlp_scripts/pipeline.sh"
    CONFIG="config/allennlp/en_gold_nontok/"
    EXPS="experiments/allennlp/en_gold_nontok/"
    
    echo "Training model on gold data..." ;sleep 1

    mkdir -p $EXPS
    $PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ normal en
    echo "Fine-tuning model on gold data..." ;sleep 1
    CONFIG="config/allennlp/en_goldsilv_nontok_tune/" #only difference from old config file: train_data_path points to gold data (as opposed to gold+silver)
fi

$PIPELINE ${CONFIG}/bert.json ${EXPS}/bert/ fine en


echo
echo "If you see this, the experiments did not throw any errors"
