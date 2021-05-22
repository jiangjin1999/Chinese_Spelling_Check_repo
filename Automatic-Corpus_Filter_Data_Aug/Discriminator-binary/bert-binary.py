from utils import *
from torch import nn
import torch
import random
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertEmbeddings, BertEncoder
import transformers as tfs
import warnings
import math
# ------------preparation----------------
random.seed(1)
warnings.filterwarnings('ignore')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# -------------train data----------------
data_right_13, data_wrong_13 = read_train("data/sighan/sighan13-train.txt")
data_right_14, data_wrong_14 = read_train("data/sighan/sighan14-train.txt")
data_right_15, data_wrong_15 = read_train("data/sighan/sighan15-train.txt")
data_right_auto, data_wrong_auto = read_train("data/Auto_Gener_Data/train.txt")
# ------------ test data--------------
data_right_test, data_wrong_test = read_test("data/sighan/sighan15-test.txt")

# ------------ raw data process -------------
print("the hybrid auto generate data(right&wrong) is "+str(len(data_right_auto))+'-'+str(len(data_wrong_auto)))
data_right_auto, data_wrong_auto = data_right_auto[0:10000], data_wrong_auto[0:10000]
# whole_data = data_right_13+data_wrong_13+data_right_14+data_wrong_14+data_right_15+data_wrong_15
whole_data = data_right_auto+data_wrong_auto

# whole_data = data_right_13+data_wrong_13+data_right_14+data_wrong_14+data_right_15+data_wrong_15+data_right_auto+data_wrong_auto
# whole_data = data_right_13+data_wrong_13

random.shuffle(whole_data)
whole_data = whole_data[math.ceil(0.2 * len(whole_data)):]

train_data = whole_data[0:math.ceil(0.9 * len(whole_data))]
dev_data = whole_data[math.ceil(0.9 * len(whole_data)): len(whole_data)]
test_data = data_wrong_test+data_right_test

train_inputs, train_targets = data_process(train_data, "train")
dev_inputs, dev_targets = data_process(dev_data, "dev")
test_inputs, test_targets = data_process(test_data, "test")

# ---------- raw train data batch -------
batch_size = 5
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
print_every_batch = 50
bert_classifier_model = BertClassificationModel()
# bert_classifier_model = torch.nn.DataParallel(model, device_ids=[0, 1])
bert_classifier_model.to(device)
# optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.9)
params = bert_classifier_model.parameters()
optimizer = torch.optim.Adam(params,
                             lr=2e-6,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=False)
criterion = nn.CrossEntropyLoss()
loss_change = []
global_f1 = 0.0
for epoch in range(epochs):
    bert_classifier_model.train()
    print_avg_loss = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        labels = torch.tensor(batch_train_targets[i])
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print_avg_loss += loss.item()
        if i % print_every_batch == (print_every_batch - 1):
            print("epoch: %d, Batch: %d, Loss: %.4f" % ((epoch + 1), (i + 1), print_avg_loss / print_every_batch))
            loss_change.append(str(print_avg_loss / print_every_batch))  # 收集loss曲线的数据
            print_avg_loss = 0

    bert_classifier_model.eval()
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
                outputs = bert_classifier_model([dev_inputs[i]])
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
            print("F1 is {}".format(str(f1)))
            if f1 > global_f1:
                # save_model(bert_classifier_model, "21号下午+auto-" + str(acc_num) + "-" + str(f1) + "-" + str(precision) + "-" + str(recall))
                global_f1 = f1

    bert_classifier_model.eval()
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
                outputs = bert_classifier_model([test_inputs[i]])
                predicted = torch.max(outputs, 1)
                predicted_list.append(int(predicted.indices))
            y_true, y_pred = test_targets, predicted_list
            TP, FN, FP, TN = perf_measure(y_true, y_pred)
            correct_num = TP
            total_num = TP + FN
            prediction_num = TP + FP
            acc_num = (TP + TN)/(TP+FN+FP+TN)
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


