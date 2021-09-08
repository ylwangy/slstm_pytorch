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

参数量 30005x(h+1) + 514xh + (28+7+6)xhxh+(7+2+1)xh+10x2xh +2xh（+hxh+h+2xh）

h=1792/1280: 186M/106M

# fine-tune

clue
| model  | 参数    | TNEWS  | IFLYTEK | WSC2020   | AFQMC   | CMNLI | CSL  | CMRC | CHID | C3 | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
|bert-base-official|108M|56.09|60.37|77.63 | 74.16| 79.47| 79.63| 85.48/64.77 | 82.20| 65.70|
|roberta-wwm-ext-official|108M|57.51|60.8|85.53| 74.30| 80.70| 80.67| 87.28/67.89 | 83.78| 67.06|
|bert-base-our-run|108M|56.33(2e-5,16,3)|59.6(2e-5,16,3)|76.97(2e-5,8,10) | 73.8(2e-5,16,3)| 80.0(3e-5,16,3)| 80.5(1e-5,8,3)| - | -| -|
|roberta-wwm-ext-our-run|108M|57.64(2e-5,16,3)|59.36(2e-5,16,3)|84.8(2e-5,8,10) |74.3(2e-5,16,3)| 81.1(3e-5,16,3)| 80.8(1e-5,8,3)| - | -| -|
|roberta-large-official|334M|57.86|62.55|-| 74.02| 81.70| 81.36| 88.61/69.94 | 85.31| 67.79|
|slstm1792-ckpt60|186M|-|-|-| -| -| -| - | -| -|

glue
| model  | 参数    | MNLI  | QNLI | QQP   | RTE   | SST | MRPC  | CoLA | STS | WNLI | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |


QA
| model  | 参数    | SQUAD 1.1  | SQUAD 2.0 | RACE | SWAG |
| :----:| :----: | :----: | :----: |:----: | :----: |

superglue
| model  | 参数    | WSC  | BoolQ | COPA   | CB   | RTE | WiC  | ReCoRD | MultiRC  | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |