tokenizer 为 Sentencepiece 模型 https://github.com/google/sentencepiece

giga_wiki_news_cn_nospecial.model 及 wiki_books_openweb_nospecial.model 为学到的tokenizer模型

vocab_cn.txt 及 vocab_en.txt 为学到的30k中英文词表 

使用方法:

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('/nfs/users/wangyile/sentencepiece/build/src/giga_wiki_news_cn_nospecial.model')

sent_ids=sp.encode('北京是中国的首都') 
print(sent_ids)

sp = spm.SentencePieceProcessor()
sp.Load('/nfs/users/wangyile/sentencepiece/build/src/wiki_books_openweb_nospecial.model')

sent_ids=sp.encode('Beijing is the captial of China') 
print(sent_ids)