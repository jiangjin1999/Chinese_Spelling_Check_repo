
from torch import nn
import torch

from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertEmbeddings, BertEncoder
import transformers as tfs
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# -------------- model ---------------------
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
model.to(device)

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-chinese')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # 嵌入层BertEmbeddings().
        self.embeddings = BertEmbeddings(config)
        # 多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.
        self.encoder = BertEncoder(config)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
    

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           pad_to_max_length=True,padding='max_length')  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        self.mask_embeddings = self.embeddings.word_embeddings.weight[103]
        dropout_output = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(dropout_output)
        return linear_output




