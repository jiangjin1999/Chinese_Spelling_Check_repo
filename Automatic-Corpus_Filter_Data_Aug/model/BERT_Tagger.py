# -*- coding:utf-8 -*-
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertEmbeddings, BertEncoder
import transformers as tfs

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)



class BertTagger(nn.Module):
    def __init__(self, tagset_size=2):
        super(BertTagger, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-chinese')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # 嵌入层BertEmbeddings().
        self.embeddings = BertEmbeddings(config)
        # 多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.
        self.encoder = BertEncoder(config)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dense = nn.Linear(768, tagset_size)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类
        self.dropout = nn.Dropout(p=0.1)  # dropout训练

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        dropout_output = self.dropout(bert_cls_hidden_state)
        tag_space = self.dense(dropout_output)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores