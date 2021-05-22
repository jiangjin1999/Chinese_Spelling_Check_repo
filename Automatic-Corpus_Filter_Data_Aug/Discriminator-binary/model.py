# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:21:39 2020

@author: Administrator
"""
import re
import torch
import random
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertEmbeddings, BertEncoder
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
import time
import os
import copy
import math

random.seed(2)
warnings.filterwarnings('ignore')
os.chdir('D:/ajiangj/exp：2---learning')
path = os.getcwd()


time_start = time.time()
file_sentence = open(path + '/merged/TrainingInputAll.txt', encoding='utf-8', errors='ignore')
file_data_sentence = file_sentence.readlines()
# example: '(sighan13-id=1)\t有一天的晚上，大家都睡得很安祥，突然一阵巨晃，我们家书贵倒了，这次地震惊动了全国。\n'
print("训练集句子的个数为：", len(file_data_sentence))
file_true = open(path + '/merged/TrainingTruthAll.txt', encoding='utf-8', errors='ignore')
file_data_true = file_true.readlines()
# '(sighan13-id=1), 28, 柜\n'
print(len(file_data_true))

data_right = []
data_wrong = []
data = []

bad_data = []

for i in range(len(file_data_sentence)):
    inp = file_data_sentence[i]
    trth = file_data_true[i]
    inp_fields = inp.strip().split("\t")
    trth_fields = trth.strip().split(", ")
    text_a = inp_fields[1]
    trth_fields_wrong_word = copy.deepcopy(trth_fields)
    for j in range(1, len(trth_fields), 2):  # 得到正确的句子
        index = int(trth_fields[j])
        right_word = trth_fields[j + 1]
        wrong_word = text_a[index - 1]
        trth_fields_wrong_word[j + 1] = wrong_word
        text_a = re.sub(text_a[index - 1], right_word, text_a)
    right_sentence_temp = text_a
    for k in range(1, len(trth_fields), 2):  # 一句话一个错词
        temp_list_right = []
        temp_list_wrong = []
        index = int(trth_fields[k])
        right_word = trth_fields[k + 1]
        wrong_word = trth_fields_wrong_word[k + 1]
        temp_right_sentences = right_sentence_temp
        temp_wrong_sentences = re.sub(right_sentence_temp[index - 1], wrong_word, right_sentence_temp)
        temp_list_right.append(temp_right_sentences)
        temp_list_right.append(1)
        temp_list_right.append(index)
        temp_list_right.append(right_word)
        temp_list_right.append(wrong_word)
        data.append(temp_list_right)

        temp_list_wrong.append(temp_wrong_sentences)
        temp_list_wrong.append(0)
        temp_list_wrong.append(index)
        temp_list_wrong.append(right_word)
        temp_list_wrong.append(wrong_word)
        data.append(temp_list_wrong)
        if right_word == wrong_word:
            bad_data.append(temp_list_right)
            bad_data.append(temp_list_wrong)
# 加上对的和错的，一共2432个句子，
# 前面的是正确的字，后面的是错误的字
# 1是正确的，0是错误的
train_data = data

test_sentence = open(path + '/merged/TestInput.txt', encoding='utf-8', errors='ignore')
test_data_sentence = test_sentence.readlines()
# example: '(pid=B2-4252-8)\t如果老师一开始用几个方法看小孩子，老师们会利用一个很简单的方法，他们不会了解甚么时候小孩子可以开始面对功课，了解礼貌的行为或是甚么东西是不对的。\n'
print("测试集句子的个数为：", len(test_data_sentence))
test_true = open(path + '/merged/TestTruth.txt', encoding='utf-8', errors='ignore')
test_data_true = test_true.readlines()
# 'B2-4252-2, 3, 现\n'
test_data = []
for i in range(len(test_data_sentence)):
    temp_list = []
    inp = test_data_sentence[i]
    trth = test_data_true[i]
    inp_fields = inp.strip().split("\t")
    trth_fields = trth.strip().split(", ")
    text_b = inp_fields[1]
    flag = 1
    if len(trth_fields) == 2:
        flag = 1
    else:
        flag = 0
    right_word_info = trth_fields[1:len(trth_fields)]
    temp_list.append(text_b)
    temp_list.append(flag)
    temp_list.append(right_word_info)
    test_data.append(temp_list)

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

model.to(device)
# test
# inputtext = "今天心情情很好啊，买了很多东西，我特别喜欢，终于有了自己喜欢的电子产品，这次总算可以好好学习了"
# tokenized_text=tokenizer.encode(inputtext)
# input_ids=torch.tensor(tokenized_text).view(-1,len(tokenized_text))
# input_ids=input_ids.to(device)
# outputs=model(input_ids)
##outputs[0].shape
##Out[145]: torch.Size([1, 49, 768])-- embedding dim 是768
##outputs[1].shape
##torch.Size([1, 768])
##对应字向量表示和句向量表示
# outputs[0].shape,outputs[1].shape
###

###
# 二、 微调
import torch
from torch import nn
from torch import optim
import transformers as tfs
# import math
import time


def series2int(series):
    length = len(series)
    a = []
    for i in range(length):
        temp = int(series[i])
        a.append(temp)
    return a


def series2list(series):
    length = len(series)
    a = []
    for i in range(length):
        temp = str(series[i])
        a.append(temp)
    return a




def train_test_val_split(df, ratio_train, ratio_test, ratio_val):
    train, middle = train_test_split(df, test_size=1 - ratio_train)
    ratio = ratio_val / (1 - ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    cla_index = []
    TP_index, FP_index, FN_index, TN_index = [], [], [], []
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
            TP_index.append(i)
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
            FP_index.append(i)
        if y_true[i] == 0 and y_pred[i] == 0:
            FN += 1
            FN_index.append(i)
        if y_true[i] == 1 and y_pred[i] == 0:
            TN += 1
            TN_index.append(i)
    cla_index.append([TP_index, FP_index, TN_index, FN_index])
    return TP, FP, TN, FN, cla_index


def evaluate_Model(inputs, targets):
    predicted_list = []
    with torch.no_grad():
        for i in range(len(inputs)):
            outputs = bert_classifier_model([inputs[i]])
            predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted.indices))
    y_true, y_pred = targets, predicted_list
    TP, FP, TN, FN, cla_index = perf_measure(y_true, y_pred)
    print("正确率是：", (TP + FN) / (TP + FP + TN + FN))
    return (TP + FN) / (TP + FP + TN + FN), TP, FP, TN, FN, cla_index


def evaluate_Model_list(inputs, targets):
    predicted_list = []
    with torch.no_grad():
        for i in range(len(inputs)):
            outputs = bert_classifier_model([inputs[i]])
            predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted.indices))
    y_true, y_pred = targets, predicted_list
    TP, FP, TN, FN, cla_index = perf_measure(y_true, y_pred)
    print("正确率是：", (TP + FN) / (TP + FP + TN + FN))
    return (TP + FN) / (TP + FP + TN + FN), TP, FP, TN, FN, cla_index, y_true, y_pred


random.shuffle(train_data)
train_list, test_list = train_data[math.ceil(0.5 * len(train_data)):math.ceil(0.6 * len(train_data))], test_data

# random.shuffle(test)

train, test = DataFrame(train_list), DataFrame(test_list)
train_inputs, train_targets = train[0].values, train[1].values
test_inputs, test_targets = test[0].values, test[1].values

print("Train set shape:", train_inputs.shape)
print(train[1].value_counts())  # 查看数据集中标签的分布
print("test set shape:", test_inputs.shape)
print(test[1].value_counts())  # 查看数据集中标签的分布

train_inputs, test_inputs = series2list(train_inputs), series2list(test_inputs)
train_targets, test_targets = series2int(train_targets), series2int(test_targets)

batch_size = 16
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_inputs[i * batch_size: (i + 1) * batch_size])
    batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])

    # 51的时候已经到0.0007
# 之后还是要，根据，把代码调成，根据他的dev1的loss去判断
# train the model


ans_test.writelines("  loss值记录：")
ans_test.writelines(str([loss_change]))
ans_test.close()

time_end = time.time()
print('totally time cost', time_end - time_start)


def list_avg_distance(list):
    sum = 0
    for i in range(len(list)):
        sum = sum + len(list[i][0])
    return sum / len(list)


dev_inputs, dev_targets = test_inputs, test_targets
bert_classifier_model = torch.load('D:/ajiangj/exp：2---learning/model：100%/twinning_test_sighan16.pkl')
bert_classifier_model.to(device)
acc_dev_s100, TP_dev_s100, FP_dev_s100, TN_dev_s100, FN_dev_s100, cla_index_dev_s100, y_true_s100, y_pre_s100 = evaluate_Model_list(
    dev_inputs, dev_targets)
index_TP_s100 = cla_index_dev_s100[0][0]
index_FP_s100 = cla_index_dev_s100[0][1]
index_TN_s100 = cla_index_dev_s100[0][2]
index_FN_s100 = cla_index_dev_s100[0][3]
sentence_TP_s100 = [test_data[i] for i in index_TP_s100]
# 30.36938775510204
sentence_FP_s100 = [test_data[i] for i in index_FP_s100]
# 31.048780487804876
sentence_TN_s100 = [test_data[i] for i in index_TN_s100]
# 33.8
sentence_FN_s100 = [test_data[i] for i in index_FN_s100]
# 30.405152224824356

bert_classifier_model = torch.load('D:/ajiangj/exp：1---pair-v1/model：5%/twinning_test_sighan15.pkl')
bert_classifier_model.to(device)
acc_dev_p005, TP_dev_p005, FP_dev_p005, TN_dev_p005, FN_dev_p005, cla_index_dev_p005, y_true_p005, y_pre_p005 = evaluate_Model_list(
    dev_inputs, dev_targets)
index_TP_p005 = cla_index_dev_p005[0][0]
index_FP_p005 = cla_index_dev_p005[0][1]
index_TN_p005 = cla_index_dev_p005[0][2]
index_FN_p005 = cla_index_dev_p005[0][3]
sentence_TP_p005 = [test_data[i] for i in index_TP_p005]
# 30.073752711496745
sentence_FP_p005 = [test_data[i] for i in index_FP_p005]
# 30.263157894736842
sentence_TN_p005 = [test_data[i] for i in index_TN_p005]
# 34.21348314606742
sentence_FN_p005 = [test_data[i] for i in index_FN_p005]
# 30.678100263852244


bert_classifier_model = torch.load('D:/ajiangj/exp：2---learning/model：5%/twinning_test_sighan15.pkl')
bert_classifier_model.to(device)
acc_dev_s005, TP_dev_s005, FP_dev_s005, TN_dev_s005, FN_dev_s005, cla_index_dev_s005, y_true_s005, y_pre_s005 = evaluate_Model_list(
    dev_inputs, dev_targets)
index_TP_s005 = cla_index_dev_s005[0][0]
index_FP_s005 = cla_index_dev_s005[0][1]
index_TN_s005 = cla_index_dev_s005[0][2]
index_FN_s005 = cla_index_dev_s005[0][3]
sentence_TP_s005 = [test_data[i] for i in index_TP_s005]
# 29.98
sentence_FP_s005 = [test_data[i] for i in index_FP_s005]
# 31.033834586466167
sentence_TN_s005 = [test_data[i] for i in index_TN_s005]
# 29.98
sentence_FN_s005 = [test_data[i] for i in index_FN_s005]
# 30.095070422535212

# read & write txt
