# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:51:46 2021

# Sentence_similarity_test_demo

@author: Administrator
"""
import torch
import transformers as tfs
from transformers import BertModel, BertTokenizer, BertConfig


model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-chinese')
bert = model_class.from_pretrained(pretrained_weights)


def CLS_calculate(sentences):
    tokenized = tokenizer.batch_encode_plus(sentences, add_special_tokens=True,
                                                           pad_to_max_length=True,padding=True)  # tokenize、add special token、pad
    input_ids = torch.tensor(tokenized['input_ids'])
    attention_mask = torch.tensor(tokenized['attention_mask'])
    bert_output = bert(input_ids, attention_mask=attention_mask)
    bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
    return bert_cls_hidden_state

        
sentence_a = '我爱你'
CLS_a = CLS_calculate(sentence_a)
print('词向量-'+str(CLS_a.shape))
sentence_b = '我喜欢你'
CLS_b = CLS_calculate(sentence_b)
print('词向量-'+str(CLS_b.shape))
