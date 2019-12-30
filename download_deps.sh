#!/bin/bash
mkdir -p data
cd data

# LM scores
wget https://nlp.stanford.edu/data/robinjia/jia2019_cert_lm_scores.zip
unzip jia2019_cert_lm_scores.zip
rm jia2019_cert_lm_scores.zip

# GloVe
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd -

# Counterfitted vectors
wget https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt.zip

# IMDB
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
rm -f aclImdb_v1.tar.gz
cp splits/imdb_train_files.txt aclImdb/train
cp splits/imdb_dev_files.txt aclImdb/train
cp splits/imdb_test_files.txt aclImdb/test

# SNLI
mkdir -p snli
cd snli 
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
mv snli_1.0/*.jsonl .
rm -r snli_1.0.zip snli_1.0 __MACOSX
