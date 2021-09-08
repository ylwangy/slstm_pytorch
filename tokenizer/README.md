tokenizer 采用 [Sentencepiece](https://github.com/google/sentencepiece) 

# 输出模型
giga_wiki_news_cn_nospecial.model 及 wiki_books_openweb_nospecial.model

# 输出词表
vocab_cn.txt 及 vocab_en.txt 


# 训练脚本

``` python
./spm_train \
    --input=INPUT_FILES \
    --model_prefix=OUTPUT_MODEL \
    --pad_id=-1 --eos_id=-1 --unk_id=0 --bos_id=-1 \
    --vocab_size=30000 \
    --character_coverage=0.999999 \
    --num_threads=64 \
    --shuffle_input_sentence=true \
    --input_sentence_size=30000000 \
    --train_extremely_large_corpus=true \


```

# 使用方法:

``` python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('./giga_wiki_news_cn_nospecial.model')

sent_tokens=sp.encode('北京是中国的首都', out_type=str)
print(sent_tokens)# '▁北京', '是中国', '的', '首都'

sent_ids=sp.encode('北京是中国的首都') 
print(sent_ids)# 4450, 3378, 3, 2375

sp = spm.SentencePieceProcessor()
sp.Load('./wiki_books_openweb_nospecial.model')

sent_tokens=sp.encode('Beijing is the captial of China', out_type=str)
print(sent_tokens)# '▁Beijing', '▁is', '▁the', '▁cap', 't', 'ial', '▁of', '▁China'

sent_ids=sp.encode('Beijing is the captial of China') 
print(sent_ids) # 6485, 12, 1, 2374, 17, 1417, 6, 706
```