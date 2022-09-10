Pytorch version of [sentence-state LSTM](https://aclanthology.org/P18-1030.pdf) and use it for [language model pre-training](https://arxiv.org/pdf/2209.03834.pdf)


# Requirements and Installation

* fairseq
* transformers
* sentencepiece


# Step

1. we follow [fairseq](https://github.com/pytorch/fairseq) for pre-training language model

2. put files in models to fairseq/fairseq/models

3. pip install . 

4. preprocess dataset and run train.sh for lm pre-training


# Checkpoints and usage

see corresponding directories


# Acknowledgement

Linyi Yang, Zhiyang Teng
