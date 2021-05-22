#-*- coding:utf-8 -*-
import torch
import torch.utils.data as data
from torch.autograd import Variable
import logging
import codecs
import random
import pickle
from bs4 import BeautifulSoup
from model.discrimitor_test_demo_2 import BertClassificationModel
import re

UNK_token=0
PAD_token=1
EOS_token=2
SOS_token=3

import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')


if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False


class Lang:
    def __init__(self):
        self.word2index = {}
        self.tag2index = {}
        self.word2count = {}
        self.tag2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.index2tag = {}
        self.n_words = 4  # Count default tokens
        self.n_tags = 0  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_tags(self, sentence):
        for word in sentence.split(' '):
            self.index_tag(word)


    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_tag(self, word):
        if word not in self.tag2index:
            self.tag2index[word] = self.n_tags
            self.tag2count[word] = 1
            self.index2tag[self.n_tags] = word
            self.n_tags += 1
        else:
            self.tag2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)  # change to index
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)  #
        return src_seq, trg_seq, self.src_seqs[index], self.trg_seqs[index]

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if (trg):
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            sequence = torch.Tensor(sequence)
        else:
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            sequence = torch.Tensor(sequence)
        return sequence


def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, src_plain, trg_plain = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, None)
    trg_seqs, trg_lengths = merge(trg_seqs, None)

    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths,  src_plain, trg_plain


def read_langs(file_name):

    logging.info(("Reading lines from {}".format(file_name)))
    total_data=[]

    with codecs.open(file_name, "r", "utf-8") as file:

        data = file.read()
        # data = data[0:2116]
        soup = BeautifulSoup(data, 'html.parser')
        results = soup.find_all('sentence')
        for item in results:

            text = item.find("text").text.strip()
            mistakes = item.find_all("mistake")

            locations = []
            for mistake in mistakes:
                location = mistake.find("location").text.strip()
                wrong =  mistake.find("wrong").text.strip()
                locations.append(int(location))
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")

            sen = list(text)
            tags = ["0" for _ in range(len(sen))]

            for i in locations:
                tags[i - 1] = "1"
            total_data.append([" ".join(sen), " ".join(tags)])

    return total_data

def read_lang_with_processed_data(file_name):
    print(("Reading from {}".format(file_name)))
    total_data = []
    with open(file_name, "r", encoding="utf-8") as f:
        data_read = f.readlines()
        for i in range(0, len(data_read), 2):
            text = data_read[i].strip()
            mistakes = data_read[i + 1].strip().split(";")
            locations = []
            for i in range(0,len(mistakes)-1):
                mistake = mistakes[i].strip().split(",")
                location = mistake[0]
                wrong = mistake[1]
                right = mistake[2]
                locations.append(int(location))
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")
            sen = list(text)
            tags = ["0" for _ in range(len(sen))]

            for i in locations:
                tags[i - 1] = "1"
            total_data.append([" ".join(sen), " ".join(tags)])
    return total_data



def get_seq(pairs,lang,batch_size,type):
    x_seq = []
    y_seq = []
    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        if(type):
            lang.index_words(pair[0])
            lang.index_tags(pair[1])

    dataset = Dataset(x_seq, y_seq,lang.word2index, lang.tag2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(filename, lang, isSplit=False, batch_size=64):

    if isSplit:

        data_path = filename
        data = read_langs(data_path)

        max_train_len = max([len(d[0].split(' ')) for d in data])
        logging.info("Number: {} and  Maxlen: {}".format(len(data), max_train_len))

        data = get_seq(data, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))
        return data

    else:
        '''
        data_path = filename
        total_data = read_lang_with_processed_data(data_path)
        random.seed(2021)

        random.shuffle(total_data)
        # ----- 去掉长度大于510 的句子

        data_510 = []
        for i in range(0, len(total_data)):
            if len(total_data[i][0])<511:
                data_510.append(total_data[i])
        total_data = data_510
        '''

        # ----- 改进：进行生成句子过滤 ------------
        # total_data_right, total_data_wrong = Discrimitor(total_data)
        # filename_total_data = save_variable(total_data, 'total_data.txt')
        # filename1 = save_variable(total_data_right, 'total_data_right.txt')
        # filename2 = save_variable(total_data_wrong, 'total_data_wrong.txt')
        #total_data_right = load_variavle('total_data_right.txt')
        #total_data_wrong = load_variavle('total_data_wrong.txt')
        #total_data = load_variavle('total_data.txt')
        total_data = load_variavle('total_data_519.txt')
        # the len of right data is 49443
        # the len of wrong data is 221800
        #print("the len of right data is "+str(len(total_data_right)))
        #print("the len of wrong data is " + str(len(total_data_wrong)))
        print("the len of total data is " + str(len(total_data)))
        #random.seed(519)
        #random.shuffle(total_data)   # condition whether to random the data.
        # total_data = total_data_wrong[0:20000]
        total_data = total_data[0:3000]

        print("the len of total input"+"5000-front and back random"+" data is " + str(len(total_data)))
        train = total_data[:int(len(total_data)*0.9)]
        dev = total_data[int(len(total_data)*0.9):]
        #test = total_data[int(len(total_data) * 0.9):]

        max_train_len = max([len(d[0].split(' ')) for d in train])
        max_dev_len = max([len(d[0].split(' ')) for d in dev])
        #max_test_len = max([len(d[0].split(' ')) for d in test])

        logging.info("Train: Number: {} and  Maxlen: {}".format(len(train), max_train_len))
        logging.info("Dev: Number: {} and  Maxlen: {}".format(len(dev), max_dev_len))
        #logging.info("Test: Number: {} and  Maxlen: {}".format(len(test),max_test_len))

        train = get_seq(train, lang, batch_size, True)
        dev = get_seq(dev, lang, batch_size, True)
        #test = get_seq(test, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))

        return train, dev

def Discrimitor(data):
    data_wrong = []
    data_right = []
    classifier_model = BertClassificationModel()
    # bert_classifier_model = torch.nn.DataParallel(model, device_ids=[0, 1])

    classifier_model = torch.load(
        'D:/Chinese Spelling Error Check coding review/Automatic-Corpus-Generation-master/Discriminator-binary/save/11号晚-0.9049751243781095-0.9043565348022032-0.8809756097560976-0.9290123456790124-model.pkl')
    # classifier_model = torch.load('D:/ajiangj/exp：1---pair-v1/twinning_test_sighan99.pkl')
    classifier_model.eval()
    classifier = classifier_model

    for i in range(0,len(data)):
        outputs = classifier([data[i][0]])
        predicted = torch.max(outputs, 1)
        label = int(predicted.indices)
        if int(label) == 0:
            data_wrong.append(data[i])
        elif int(label) == 1:
            data_right.append(data[i])

    return data_right,data_wrong

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def old_prepare_data_seq(filename, lang, isSplit=False, batch_size=64):

    if isSplit:

        data_path = filename
        data = read_langs(data_path)

        max_train_len = max([len(d[0].split(' ')) for d in data])
        logging.info("Number: {} and  Maxlen: {}".format(len(data), max_train_len))

        data = get_seq(data, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))
        return data

    else:
        data_path = filename
        total_data = read_langs(data_path)

        random.shuffle(total_data)

        train = total_data[:int(len(total_data)*0.9)]
        dev = total_data[int(len(total_data)*0.9):]
        #test = total_data[int(len(total_data) * 0.9):]

        max_train_len = max([len(d[0].split(' ')) for d in train])
        max_dev_len = max([len(d[0].split(' ')) for d in dev])
        #max_test_len = max([len(d[0].split(' ')) for d in test])

        logging.info("Train: Number: {} and  Maxlen: {}".format(len(train), max_train_len))
        logging.info("Dev: Number: {} and  Maxlen: {}".format(len(dev), max_dev_len))
        #logging.info("Test: Number: {} and  Maxlen: {}".format(len(test),max_test_len))

        train = get_seq(train, lang, batch_size, True)
        dev = get_seq(dev, lang, batch_size, True)
        #test = get_seq(test, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))

        return train, dev


# ----------------- data augmentation ------------------
def read_lang_with_processed_data_with_data_augmentation(file_name):
    print(("Reading from {}".format(file_name)))
    total_data = []
    with open(file_name, "r", encoding="utf-8") as f:
        data_read = f.readlines()
        for i in range(0, len(data_read), 2):
            sentence_tmp = []
            text = data_read[i].strip()
            mistakes = data_read[i + 1].strip().split(";")
            locations = []
            right_sentence = text
            for i in range(0,len(mistakes)-1):
                mistake = mistakes[i].strip().split(",")
                location = mistake[0]
                wrong = mistake[1]
                right = mistake[2]
                right_sentence = replace_char(right_sentence,right,int(location)-1)
                locations.append(int(location))
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")
            sen = list(text)
            sen_right = list(right_sentence)
            tags_right = ["0" for _ in range(len(sen))]
            tags = ["0" for _ in range(len(sen))]
            for i in locations:
                tags[i - 1] = "1"
            temp_right_sent = right_sentence

            for i in range(0,len(mistakes)-1):
                temp_tag = ["0" for _ in range(len(sen))]
                temp_mis = mistakes[i].strip().split(",")
                temp_location = temp_mis[0]
                temp_wrong = temp_mis[1]
                # temp_right = temp_mis[2]
                temp_sent= replace_char(temp_right_sent,temp_wrong,int(temp_location)-1)
                temp_tag[int(temp_location)-1]='1'
                sentence_tmp.append([" ".join(temp_sent), " ".join(temp_tag)])
            if int(len(mistakes)) > 2:
                sentence_tmp.append([" ".join(sen), " ".join(tags)])
            sentence_tmp.append([" ".join(sen_right), " ".join(tags_right)])
            total_data.append(sentence_tmp)
    return total_data


def read_lang_with_filter_data_with_data_augmentation(file_name):
    print(("Reading from {}".format(file_name)))
    total_data = []
    with open(file_name, "r", encoding="utf-8") as f:
        data_read = f.readlines()
        for i in range(0, len(data_read), 2):
            sentence_tmp = []
            text = data_read[i].strip()
            mistakes = data_read[i + 1].strip().split(";")
            locations = []
            right_sentence = text
            for i in range(0,len(mistakes)-1):
                mistake = mistakes[i].strip().split(",")
                location = mistake[0]
                wrong = mistake[1]
                right = mistake[2]
                right_sentence = replace_char(right_sentence,right,int(location)-1)
                locations.append(int(location))
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")
            sen = list(text)
            sen_right = list(right_sentence)
            tags_right = ["0" for _ in range(len(sen))]
            tags = ["0" for _ in range(len(sen))]
            for i in locations:
                tags[i - 1] = "1"
            temp_right_sent = right_sentence

            for i in range(0,len(mistakes)-1):
                temp_tag = ["0" for _ in range(len(sen))]
                temp_mis = mistakes[i].strip().split(",")
                temp_location = temp_mis[0]
                temp_wrong = temp_mis[1]
                # temp_right = temp_mis[2]
                temp_sent= replace_char(temp_right_sent,temp_wrong,int(temp_location)-1)
                temp_tag[int(temp_location)-1]='1'
                sentence_tmp.append([" ".join(temp_sent), " ".join(temp_tag)])
            if int(len(mistakes)) > 2:
                sentence_tmp.append([" ".join(sen), " ".join(tags)])
            sentence_tmp.append([" ".join(sen_right), " ".join(tags_right)])
            total_data.append(sentence_tmp)
    return total_data

def replace_char(old_string, char, index):
    '''
    字符串按索引位置替换字符
    '''
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index+1:]
    return new_string

def prepare_data_seq_with_data_augmentation(filename, lang, isSplit=False, batch_size=64):

    if isSplit:

        data_path = filename
        data = read_langs(data_path)

        max_train_len = max([len(d[0].split(' ')) for d in data])
        logging.info("Number: {} and  Maxlen: {}".format(len(data), max_train_len))

        data = get_seq(data, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))
        return data

    else:

        data_path = filename
        total_data = read_lang_with_processed_data_with_data_augmentation(data_path)
        # random.seed(2021)

        # random.shuffle(total_data)
        # ----- 去掉长度大于510 的句子
        '''
        data_510 = []
        for i in range(0, len(total_data)):
            if len(total_data[i][0])<511:
                data_510.append(total_data[i])
        total_data = data_510
        

        # ----- 改进：进行生成句子过滤 ------------
        # total_data_right, total_data_wrong = Discrimitor(total_data)
        # filename_total_data = save_variable(total_data, 'total_data.txt')
        # filename1 = save_variable(total_data_right, 'total_data_right.txt')
        # filename2 = save_variable(total_data_wrong, 'total_data_wrong.txt')
        '''
        #total_data_right = load_variavle('total_data_right.txt')
        #total_data_wrong = load_variavle('total_data_wrong.txt')
        #total_data = load_variavle('total_data.txt')

        # the len of right data is 49443
        # the len of wrong data is 221800
        #print("the len of right data is "+str(len(total_data_right)))
        #print("the len of wrong data is " + str(len(total_data_wrong)))
        #print("the len of total data is " + str(len(total_data)))
        random.seed(519)
        random.shuffle(total_data)   # condition whether to random the data.
        # total_data = total_data_wrong[0:20000]
        total_data_temp = total_data[0:12000]
        #filename_temp = save_variable(total_data_temp, 'total_data_aug.txt')
        #total_data_temp = load_variavle('total_data_aug.txt')
        total_data_temp = total_data_temp[0:12000]
        data_temp=[]
        for i in range(0,int(len(total_data_temp))):
            for j in range(0,int(len(total_data_temp[i]))):
                data_temp.append(total_data_temp[i][j])
        total_data = data_temp

        random.shuffle(total_data)
        train = total_data[:int(len(total_data)*0.9)]
        dev = total_data[int(len(total_data)*0.9):]
        #test = total_data[int(len(total_data) * 0.9):]

        max_train_len = max([len(d[0].split(' ')) for d in train])
        max_dev_len = max([len(d[0].split(' ')) for d in dev])
        #max_test_len = max([len(d[0].split(' ')) for d in test])

        logging.info("Train: Number: {} and  Maxlen: {}".format(len(train), max_train_len))
        logging.info("Dev: Number: {} and  Maxlen: {}".format(len(dev), max_dev_len))
        #logging.info("Test: Number: {} and  Maxlen: {}".format(len(test),max_test_len))

        train = get_seq(train, lang, batch_size, True)
        dev = get_seq(dev, lang, batch_size, True)
        #test = get_seq(test, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))

        return train, dev