# Neural DRS Parsing, with alignment
This repository contains code for extending the DRS parser of van Noord et al. (https://github.com/RikVN/Neural_DRS/) such that is also produces the alignment between words of the input sequence and DRS-clauses which is found in the PMB data. The code of the original Neurla DRS repository is modified, however, since some file that my work modifies are not directly present in the original repository but rather downloaded through a setup script, my modification of their work is not a fork but instead its own repository with a patch script. To get it running, https://github.com/RikVN/Neural_DRS/ must first be cloned and set up and, following that, this repository must be cloned to the same directory and the patch script applied. Detailed instructions are provided further below.

Modifications are made to preprocessing, postprocessing, evaluation, shell-scripts as well as allenNLP code:
* Pre- and postprocessing are modified to expected the alignment alongside a DRS-clause. Whereas previously the alignment was discarded and only the DRS-clause considered, now both a passed through the program, though processing still mostly takes place on the DRS-clause.
* Evaluation is also changed to expect alignment and expanded to be able to evaluate the alignment on top of pure DRS parsing.
* AllenNLP code is modified to make it possible to extract the desired alignment from the attention mechanism.

To run, first setup Neural DRS as instructed in the repository (EMNLP paper) until you can successfully run the unit tests. Then clone this repository and and run patch.sh. Here is a full list of commands I had to run to set everything up, including some fixes:

```
git clone https://github.com/RikVN/Neural_DRS
cd Neural_DRS
cur_dir=$(pwd)
export PYTHONPATH=${cur_dir}/DRS_parsing/:${PYTHONPATH}
export PYTHONPATH=${cur_dir}/DRS_parsing/evaluation/:${PYTHONPATH}
conda create -n ndrs python=3.6
conda activate ndrs
./src/setup.sh
git clone https://github.com/RikVN/allennlp
cd allennlp
git checkout DRS
pip install --editable . ; cd ../
mkdir -p emb; cd emb
wget "http://www.let.rug.nl/rikvannoord/embeddings/glove_pmb.zip"
unzip glove_pmb.zip; rm glove_pmb.zip;  cd ../
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
conda install pytorch==1.3.1 torchvision cudatoolkit=9.2 -c pytorch
pip uninstall numpy
pip uninstall numpy
pip install numpy
cd allennlp
git checkout fix
cd ../
pip install overrides==3.1.0
conda install pytorch=1.10.2=py3.6_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3.1=h2bc3f7f_2 -c pytorch



https://github.com/GitianOberhuber/Neural_DRS_alignment
cd Neural_DRS_alignment/
./patch.sh
cd ..
``` 
