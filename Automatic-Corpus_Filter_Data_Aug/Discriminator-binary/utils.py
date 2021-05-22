import re
from pandas.core.frame import DataFrame
import os
import torch

def read_test(file_name):
    print(("Reading from {}".format(file_name)))
    data_right = []
    data_wrong = []
    with open(file_name,"r", encoding="utf-8") as f:
        data_read = f.readlines()
        for i in range(0, len(data_read), 2):
            temp_data_right = []
            temp_data_wrong = []
            src = data_read[i].strip()
            detail = data_read[i+1].strip().split(";")
            tgt = get_tgt_sentence(src, detail)
            # 有错误信息的是0 无错误信息的 是1
            if len(detail) >1:
                temp_data_wrong.append(src)
                temp_data_wrong.append(0)
                data_wrong.append(temp_data_wrong)
            else:
                temp_data_right.append(src)
                temp_data_right.append(1)
                data_right.append(temp_data_right)

            # 1 是正确的，0是错误的

        return data_right, data_wrong

def read_train(file_name):
    print(("Reading from {}".format(file_name)))
    data_right = []
    data_wrong = []
    with open(file_name,"r", encoding="utf-8") as f:
        data_read = f.readlines()
        for i in range(0, len(data_read), 2):
            temp_data_right = []
            temp_data_wrong = []
            src = data_read[i].strip()
            detail = data_read[i+1].strip().split(";")
            tgt = get_tgt_sentence(src, detail)
            temp_data_wrong.append(src)
            temp_data_wrong.append(0)
            data_wrong.append(temp_data_wrong)
            temp_data_right.append(tgt)
            temp_data_right.append(1)
            data_right.append(temp_data_right)
            # 1 是正确的，0是错误的

        return data_right, data_wrong

def get_tgt_sentence(src, detail):
    tgt = src
    for i in range(0, len(detail)-1):
        details = detail[i].strip().split(',')
        location = int(details[0])
        wrong_word = details[1]
        right_word = details[2]
        if tgt[location-1] == wrong_word:
            tgt = replace_char(tgt, right_word, location-1)
            # re.sub(tgt[location-1], right_word, tgt)
        else: print(tgt+details[0]+wrong_word+right_word+" message wrong")
    return tgt

def replace_char(old_string, char, index):
    '''
    字符串按索引位置替换字符
    '''
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index+1:]
    return new_string

def data_process(data, label):
    data = DataFrame(data)
    data_inputs = data[0].values
    data_target = data[1].values
    print(label+" set shape:", data_inputs.shape)
    print(data[1].value_counts())
    data_inputs = series2list(data_inputs)
    data_target = series2int(data_target)
    return data_inputs, data_target


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


def evaluate_Model(model, inputs, targets):
    correct_num = 0
    total_num = 0
    prediction_num = 0
    f1 = 0
    bert_classifier_model =model
    predicted_list = []
    with torch.no_grad():
        for i in range(len(inputs)):
            outputs = bert_classifier_model([inputs[i]])
            predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted.indices))
    y_true, y_pred = targets, predicted_list
    TP, FN, FP, TN  = perf_measure(y_true, y_pred)
    correct_num = TP
    total_num = TP + FN
    prediction_num = TP + FP
    precision = 1.0 * correct_num / prediction_num
    recall = 1.0 * correct_num / total_num
    f1 = 2 * recall * precision / (recall + precision)
    return f1, precision, recall

def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    # cla_index = []
    TP_index, FP_index, FN_index, TN_index = [], [], [], []
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
           TP_index.append(i)
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
           FP_index.append(i)
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1
           FN_index.append(i)
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1
           TN_index.append(i)
    # cla_index.append([TP_index, FP_index, TN_index, FN_index])
    return TP, FN, FP, TN

def evaluation(model, inputs, targets):
    global_f1 = 0
    f1, precision, recall = evaluate_Model(model, inputs, targets)
    directory = 'save/' + str(f1)+"-"+str(precision)+"-"+str(recall)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if f1 > global_f1:
        torch.save(model, directory + "/" + '-model.th')
        print("Recall is {}".format(str(recall)))
        print("Precision is {}".format(str(precision)))
        print("F1 is {}".format(str(f1)))
        global_f1 = f1



def save_model(model,name):
    directory = 'save/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model, directory + "/"+ name+ '-model.pkl')
