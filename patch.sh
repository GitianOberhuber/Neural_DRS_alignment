#!/bin/bash
echo "Trying to replace original Neural_DRS files..."

cd ..
if [ ! -d "src" ]; then
    echo "The directory src does not exist, make sure the original DRS_Parsing repository is present! Exiting..."
    exit 1  # Exit with a non-zero status code to indicate an error
fi
if [ ! -d "src/allennlp_scripts" ]; then
    echo "The directory src/allennlp_scripts does not exist, make sure the original DRS_Parsing repository is present! Exiting..."
    exit 1  # Exit with a non-zero status code to indicate an error
fi
if [ ! -d "DRS_parsing" ]; then
    echo "The directory DRS_parsing does not exist, make sure the original DRS_Parsing repository is present! Exiting..."
    exit 1  # Exit with a non-zero status code to indicate an error
fi
if [ ! -d "DRS_parsing/evaluation" ]; then
    echo "The directory DRS_parsing/evaluation does not exist, make sure the original DRS_Parsing repository is present! Exiting..."
    exit 1  # Exit with a non-zero status code to indicate an error
fi

cp -u -R Neural_DRS_alignment/src .
cp -u -R Neural_DRS_alignment/DRS_parsing .
cp -u -R Neural_DRS_alignment/config .
cp -u -R Neural_DRS_alignment/allennlp .



