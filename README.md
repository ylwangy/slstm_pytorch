# Requirements and Installation

* fairseq
* transformers
* sentencepiece


# Step

1.git clone [fairseq](https://github.com/pytorch/fairseq)

2.将models下slstm文件夹放在fairseq/fairseq/models下

3.pip install 配置环境 

# 使用模型

参考example

# fine-tune

clue
| model  | 参数    | TNEWS  | IFLYTEK | WSC   | CMNLI   | AFQMC | CSL  | CMRC | CHID | C3 | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
|bert-base-official|108M|56.09/56.58|60.37/60.29|59.6/62.0 | -| -| -| - | -| -|
|roberta-wwm-ext-official|108M|57.51/56.94|60.8/67.2|/67.8 | -| -| -| - | -| -|
|bert-base-our-run|108M|-|-|- | -| -| -| - | -| -|
|roberta-wwm-ext-our-run|108M|-|-|- | -| -| -| - | -| -|

glue
| model  | 参数    | MNLI  | QNLI | QQP   | RTE   | SST | MRPC  | CoLA | STS | WNLI | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |


QA
| model  | 参数    | SQUAD 1.1  | SQUAD 2.0 | RACE | SWAG |
| :----:| :----: | :----: | :----: |:----: | :----: |

superglue
| model  | 参数    | WSC  | BoolQ | COPA   | CB   | RTE | WiC  | ReCoRD | MultiRC  | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |