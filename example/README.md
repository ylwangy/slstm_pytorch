模型使用方法


``` python
import torch

import sentencepiece as spm
from fairseq.models.slstm import SlstmModel

### load sentencepiece model
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load('/nfs/users/wangyile/sentencepiece/build/src/giga_wiki_news_cn_nospecial.model')

### load lstm model
slstm = SlstmModel.from_pretrained('/nfs/users/wangyile/fairseq/fairseq/models/slstm/cn-wiki/slstm1792_ln-003_posinput-dp0/', checkpoint_file='checkpoint_last.pt')

slstm.cuda()
slstm.eval()


sent_ids=tokenizer.encode('北京是中国的首都') 

print(sent_ids)# 4450, 3378, 3, 2375

list_special_token=[0] #bos
for id_ in sent_ids:
    list_special_token.append(id_ + 4)   # bos pad eos unk 4,......,30003, mask
list_special_token.append(2) #eos

print(list_special_token)# 0, 4454, 3382, 7, 2379, 2
tokens = torch.LongTensor(list_special_token)
tokens = tokens.unsqueeze(0)


### features
h, _, _, g = slstm.model(tokens.long().cuda(),features_only=True,return_all_hiddens=False)
print(h.size()) # 1, 6, 1792
print(g.size()) # 1, 1792


### register_linear_for_g_node
slstm.model.register_classification_head('3c-task',num_classes=3)
slstm.cuda()
slstm.eval()
h, _, _, g = slstm.model(tokens.long().cuda(),features_only=True,classification_head_name='3c-task',return_all_hiddens=False)
print(h.size()) # 1, 6, 1792
print(g.size()) # 1, 3


### verify mask language model --- cn
mask_position = 4 # 北京 是中国 的 mask

list_special_token[mask_position] = 30000+4 # mask
tokens = torch.LongTensor(list_special_token)
tokens = tokens.unsqueeze(0)


h, _, _, g = slstm.model(tokens.long().cuda(),features_only=False,return_all_hiddens=False)
logits = h[0, mask_position, :].squeeze()

prob = logits.softmax(dim=0)
values, index = prob.topk(k=10, dim=0)
print(index) 
# 307,  2379,  6366,  1446,  5661,  4375, 10462,  1288,  8134,  3426
# 中心，首都， 领土,  ...

### verify mask language model --- en
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load('/nfs/users/wangyile/sentencepiece/build/src/wiki_books_openweb_nospecial.model')

slstm = SlstmModel.from_pretrained('/nfs/users/wangyile/fairseq/fairseq/models/slstm/en-wiki/slstm1792_ln-003_posinput-dp0/', checkpoint_file='checkpoint_last.pt')

slstm.cuda()
slstm.eval()


sent_ids=tokenizer.encode('Beijing is the captial of China') #'▁Beijing', '▁is', '▁the', '▁cap', 't', 'ial', '▁of', '▁China'

print(sent_ids)# 6485, 12, 1, 2374, 17, 1417, 6, 706

list_special_token=[0] #bos
for id_ in sent_ids:
    list_special_token.append(id_ + 4)   # bos pad eos unk 4,......,30003, mask
list_special_token.append(2) #eos

print(list_special_token)# 0, 6489, 16, 5, 2378, 21, 1421, 10, 710, 2
mask_position = 8 # ▁Beijing ▁is ▁the ▁cap t ial ▁of mask

list_special_token[mask_position] = 30000+4 # mask
tokens = torch.LongTensor(list_special_token)
tokens = tokens.unsqueeze(0)

h, _, _, g = slstm.model(tokens.long().cuda(),features_only=False,return_all_hiddens=False)
logits = h[0, mask_position, :].squeeze()

prob = logits.softmax(dim=0)
values, index = prob.topk(k=10, dim=0)
print(index) 
# 6489,   710,  7826,  1053,  7270,  1152, 15913,    33,  5299, 14253
# Beijing , China, Taiwan, India, ...
```