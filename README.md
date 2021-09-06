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

参数量 30005*(h+1)+514*h + (28+7+6)*h*h+(7+2+1)*h+10*2*h +2*h （+h*h+h+2*h）

h=1792 186M
h=1280 106M

# fine-tune

clue
| model  | 参数    | TNEWS  | IFLYTEK | WSC2020   | CMNLI   | AFQMC | CSL  | CMRC | CHID | C3 | 
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
|bert-base-official|108M|56.09/56.58|60.37/60.29|77.63 | -| -| -| - | -| -|
|roberta-wwm-ext-official|108M|57.51/56.94|60.8/60.3|85.53| -| -| -| - | -| -|
|bert-base-our-run|108M|56.33(2e-5,16,3)|59.6(2e-5,16,3)|76.97(2e-5,8,10) | -| -| -| - | -| -|
|roberta-wwm-ext-our-run|108M|57.64(2e-5,16,3)|59.36(2e-5,16,3)|84.8(2e-5,8,10) | -| -| -| - | -| -|
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