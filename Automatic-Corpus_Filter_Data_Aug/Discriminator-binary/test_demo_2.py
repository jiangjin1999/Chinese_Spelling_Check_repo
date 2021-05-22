from utils import *
from torch import nn
import torch
import random
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertEmbeddings, BertEncoder
import transformers as tfs
import warnings
import math
import numpy
# ------------preparation----------------
random.seed(1)
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------train data----------------
data_right_13, data_wrong_13 = read_train("data/sighan/sighan13-train.txt")
data_right_14, data_wrong_14 = read_train("data/sighan/sighan14-train.txt")
data_right_15, data_wrong_15 = read_train("data/sighan/sighan15-train.txt")
# ------------ test data--------------
data_right_test, data_wrong_test = read_test("data/sighan/sighan15-test.txt")

# ------------ raw data process -------------
whole_data = data_right_13+data_wrong_13+data_right_14+data_wrong_14+data_right_15+data_wrong_15
random.shuffle(whole_data)

train_data = whole_data[0:math.ceil(0.9 * len(whole_data))]
dev_data = whole_data[math.ceil(0.9 * len(whole_data)): len(whole_data)]
test_data = data_wrong_test+data_right_test

train_inputs, train_targets = data_process(train_data, "train")
dev_inputs, dev_targets = data_process(dev_data, "dev")
test_inputs, test_targets = data_process(test_data, "test")
dev_inputs, dev_targets = dev_inputs[0:100], dev_targets[0:100]
# ---------- raw train data batch -------
batch_size = 16
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_inputs[i * batch_size: (i + 1) * batch_size])
    batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])

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
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
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


# ---------------train model -----------------------
epochs = 20
# lr = 0.00001
print_every_batch = 20
classifier_model = BertClassificationModel()
# bert_classifier_model = torch.nn.DataParallel(model, device_ids=[0, 1])

classifier_model = torch.load('D:/Chinese Spelling Error Check coding review/Automatic-Corpus-Generation-master/Discriminator-binary/save/11号晚-0.9049751243781095-0.9043565348022032-0.8809756097560976-0.9290123456790124-model.pkl')
# classifier_model = torch.load('D:/ajiangj/exp：1---pair-v1/twinning_test_sighan99.pkl')
classifier_model.eval()
classifier_model.to(device)
print("============= Begin Validation =========================\n")
if True:
    # See what the score are after training
    acc_num = 0
    correct_num = 0
    total_num = 0
    prediction_num = 0
    predicted_list = []
    with torch.no_grad():
        for i in range(len(dev_inputs)):
            outputs = classifier_model([dev_inputs[i]])
            predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted.indices))
        y_true, y_pred = dev_targets, predicted_list
        TP, FN, FP, TN = perf_measure(y_true, y_pred)
        correct_num = TP
        total_num = TP + FN
        prediction_num = TP + FP
        acc_num = (TP + TN) / (TP + FN + FP + TN)
        precision = 1.0 * correct_num / prediction_num
        recall = 1.0 * correct_num / total_num
        f1 = 2 * recall * precision / (recall + precision)
        print("==========Validation Result ==============\n")
        print("TP is {}".format(str(TP)))
        print("FN is {}".format(str(FN)))
        print("FP is {}".format(str(FP)))
        print("TN is {}".format(str(TN)))
        print("Accuracy is {}".format(str(acc_num)))
        print("Recall is {}".format(str(recall)))
        print("Precision is {}".format(str(precision)))
        print("F1 is {}\n".format(str(f1)))



print("============= Begin Test =========================\n")
if True:
    # See what the score are after training
    acc_num = 0
    correct_num = 0
    total_num = 0
    prediction_num = 0
    predicted_list = []
    with torch.no_grad():
        for i in range(len(test_inputs)):
            outputs = classifier_model([test_inputs[i]])
            predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted.indices))
        y_true, y_pred = test_targets, predicted_list
        TP, FN, FP, TN = perf_measure(y_true, y_pred)
        correct_num = TP
        total_num = TP + FN
        prediction_num = TP + FP
        acc_num = (TP + TN) / (TP + FN + FP + TN)
        precision = 1.0 * correct_num / prediction_num
        recall = 1.0 * correct_num / total_num
        f1 = 2 * recall * precision / (recall + precision)

        print("==========Test15 Result ==============\n")
        print("TP is {}".format(str(TP)))
        print("FN is {}".format(str(FN)))
        print("FP is {}".format(str(FP)))
        print("TN is {}".format(str(TN)))
        print("Accuracy is {}".format(str(acc_num)))
        print("Recall is {}".format(str(recall)))
        print("Precision is {}".format(str(precision)))
        print("F1 is {}\n".format(str(f1)))
        
text = ['之前，我们简介了字符串相关的处理函数。我们可以通过这些函数实现简单的搜索功能，比如说从字符串“I love you”中搜索是否有“you”这一子字符串。但有些时候，我们只是模糊地知道我们想要找什么，而不能具体说出我是在找“you”，比如说，我想找出字符串中包含的数字，这些数字可以是0到9中的任何一个。这些模糊的目标可以作为信息写入正则表达式，传递给Python，从而让Python知道我们想要找的是什么。']
    
import re
text_processed=re.split(" |。|！",text[0])        

for i in range(0, len(text_processed)):
    print(text_processed[i])
    outputs = classifier_model([text_processed[i]])
    predicted = torch.max(outputs, 1)
    label = int(predicted.indices)
    print(label)
    
inputs = ['当着']    
outputs = classifier_model(inputs)
predicted = torch.max(outputs, 1)
label = int(predicted.indices)
print(label)
# 句号有一定影响--挡着
