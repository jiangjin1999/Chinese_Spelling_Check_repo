#- *- coding: utf-8 -*-
import numpy as np
import logging
from tqdm import tqdm
from utils.utils import *
from bilstm import *
import torch.optim as optim
import pickle
from BERT_Tagger import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

EMBEDDING_DIM = 300
HIDDEN_DIM = 300
batch_size = 128

isSplit = True


lang = Lang()
#train, dev = prepare_data_seq_with_data_augmentation("data/Auto_Gener_Data/train.txt", lang,  False, batch_size)
train, dev = prepare_data_seq("data/Auto_Gener_Data/train.txt", lang,  False, batch_size)
#train, dev = old_prepare_data_seq("data/train/train-cut-one_third.sgml", lang,  False, batch_size)
#train, dev = old_prepare_data_seq("data/sighan/train15.sgml", lang,  False, batch_size)
test13 = prepare_data_seq("data/sighan/test13.sgml", lang, True, batch_size)
test14 = prepare_data_seq("data/sighan/test14.sgml", lang, True, batch_size)
test15 = prepare_data_seq("data/sighan/test15.sgml", lang, True, batch_size)

model = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, lang.n_words, lang.n_tags)
# model = BertTagger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_function = nn.NLLLoss(weight=torch.Tensor([1,5]).cuda())
#optimizer = optim.RMSprop(model.parameters(), lr=0.000001)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)

with open("modeldict.pkl","wb") as file:
     pickle.dump(lang, file, pickle.HIGHEST_PROTOCOL)

global_f1 = 0.0
num_epoch = 0
for epoch in range(1000):  # again, normally you would NOT do 300 epochs, it is toy data
    num_epoch += 1
    print("The {} epoch in Training".format(epoch))
    batch_num = 0
    model.train()
    for  src_index, src_length, target_index, target_length, _, _ in train:
        batch_num += 1
        model.zero_grad()
        #　问题在于，这里的ｓｒｃ＿ｉｎｄｅｘ是符号而不是文字。这个还需要花时间去处理。　
        tag_scores = model(src_index, src_length)
        # tag_scores = model(src_index)
        target_scores = target_index.transpose(0, 1)

        total_loss = 0.0
        try:
            for i in range(len(src_length)):
                total_loss += loss_function(tag_scores[i][:src_length[i]], target_scores[i][:src_length[i]])
        except Exception as e:
            print("src:")
            print(src_index)
            print(src_length)
            print("target:")
            print(target_index)
            print(target_length)

            print("exception is:")
            print(e)


        loss = total_loss / len(src_length)
        loss.backward()

        if batch_num % 10000 == 0:
            print("The {} batch is {}".format(batch_num, loss.data.item()))
        optimizer.step()

    if num_epoch % 5 != 0:
        continue

    model.eval()
    print("============= Begin Validation =========================\n")
    if True:
        # See what the score are after training
        with torch.no_grad():
            correct_num = 0
            total_num = 0
            prediction_num = 0
            for batch_data in dev:
                model.zero_grad()
                model.hidden = model.get_state(batch_data[0])
                prediction_scores = model(batch_data[0], batch_data[1])
                prediction_scores = torch.argmax(prediction_scores, dim=2)
                target_scores = batch_data[2].transpose(0, 1)
                for idx, sen_len in enumerate(batch_data[3]):
                    for i in range(sen_len):
                        if target_scores[idx][i].item() == 1:
                            total_num += 1
                            if prediction_scores[idx][i] == target_scores[idx][i]:
                                correct_num += 1
                        if prediction_scores[idx][i].item() == 1:
                            prediction_num += 1
            precision = 1.0 * correct_num / prediction_num
            recall = 1.0 * correct_num / total_num
            f1 = 2 * recall * precision / (recall + precision)
            if f1 > global_f1:
                save_model(model, "12号早-"+str(f1)+"-"+str(precision)+"-"+str(recall))
                print("Precision is {}".format(str(precision)))
                print("Recall is {}".format(str(recall)))
                print("F1 is {}".format(str(f1)))
                glabal_f1 = f1
    print("============= Begin Test =========================\n")
    model.eval()
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        prediction_num = 0
        for batch_data in test15:
            model.zero_grad()
            model.hidden = model.get_state(batch_data[0])
            prediction_scores = model(batch_data[0], batch_data[1])
            prediction_scores = torch.argmax(prediction_scores, dim=2)
            target_scores = batch_data[2].transpose(0, 1)
            for idx, sen_len in enumerate(batch_data[3]):
                for i in range(sen_len):
                    if target_scores[idx][i].item() == 1:
                        total_num += 1
                        if prediction_scores[idx][i] == target_scores[idx][i]:
                            correct_num += 1
                    if prediction_scores[idx][i].item() == 1:
                        prediction_num += 1

        precision = 1.0 * correct_num / prediction_num
        recall = 1.0 * correct_num / total_num
        f1 = 2 * recall * precision / (recall + precision)

        print("==========Test15 Result ==============\n")
        print("Precision is {}\n".format(str(precision)))
        print("Recall is {}\n".format(str(recall)))
        print("F1 is {}\n".format(str(f1)))
'''
        print("============= Begin Test 2 =========================\n")
    model.eval()
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        prediction_num = 0
        for batch_data in test14:
            model.zero_grad()
            model.hidden = model.get_state(batch_data[0])
            prediction_scores = model(batch_data[0], batch_data[1])
            prediction_scores = torch.argmax(prediction_scores, dim=2)
            target_scores = batch_data[2].transpose(0, 1)
            for idx, sen_len in enumerate(batch_data[3]):
                for i in range(sen_len):
                    if target_scores[idx][i].item() == 1:
                        total_num += 1
                        if prediction_scores[idx][i] == target_scores[idx][i]:
                            correct_num += 1
                    if prediction_scores[idx][i].item() == 1:
                        prediction_num += 1

        precision = 1.0 * correct_num / prediction_num
        recall = 1.0 * correct_num / total_num
        f1 = 2 * recall * precision / (recall + precision)

        print("==========Test14 Result ==============\n")
        print("Recall is {}\n".format(str(recall)))
        print("Precision is {}\n".format(str(precision)))
        print("F1 is {}\n".format(str(f1)))

'''



