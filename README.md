### Pytorch version of sentence-state LSTM and use it for language model pre-training.

[Sentence-State LSTM for Text Representation](https://aclanthology.org/P18-1030.pdf)

[Pre-Training a Graph Recurrent Network for Language Representation](https://arxiv.org/pdf/2209.03834.pdf)

## Requirements and Installation

* fairseq
* transformers
* sentencepiece


## Step

1. we follow [fairseq](https://github.com/pytorch/fairseq) for pre-training language model

2. put files in models to fairseq/fairseq/models

3. pip install . 

4. preprocess dataset and run train.sh for lm pre-training


## Checkpoints and usage

see corresponding directories


## Cite

      @article{zhang2018sentence,
        title={Sentence-state lstm for text representation},
        author={Zhang, Yue and Liu, Qi and Song, Linfeng},
        journal={arXiv preprint arXiv:1805.02474},
        year={2018}
      }

      @article{wang2022pre,
        title={Pre-Training a Graph Recurrent Network for Language Representation},
        author={Wang, Yile and Yang, Linyi and Teng, Zhiyang and Zhou, Ming and Zhang, Yue},
        journal={arXiv preprint arXiv:2209.03834},
        year={2022}
      }
