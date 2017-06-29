'''
lstmを使って文章生成
5-gramに変更
batch_input_shape=(None,\
                   LSTMの中間層に入力するデータの数（※文書データなら単語の数）,\
                   LSTM中間層に投入するデータの次元数（※文書データなら１次元配列なので1)
                  )
'''


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

import numpy as np
import numpy.random as rand
import random
import sys

from mecab_test import get_words_to_katakana
from mecab_test import kata_to_hira

#import matplotlib.pyplot as plt
import pylab as plt

class DataSet():
    def __init__(self):
        self.wordmap = {}

    def setdata(self):
        wordlist = list("sあぁいぃうゔぅえぇおぉかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづってでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゃゆゅよょらりるれろわゐゑをんー?!()「」、。,.・e")
        print(wordlist)
        for value in wordlist:
            tlist = np.zeros(len(wordlist))
            tlist[wordlist.index(value)] = 1
            self.wordmap[value] = tlist

    def get_wordlist_len(self):
        return len(self.wordmap)


class ReadFile():
    def __init__(self):
        self.input_wordlist = []

    def readfile(self,fname):
        try:
            return open(fname)
        except IOError:
            print('cannot be opened.')
            sys.exit(0)

    def make_wordlist(self,fdata):
        for word in fdata.readlines():
            if word != "\n" :
                self.input_wordlist.append(word)

    def to_hiragana(self):
        self.hira_word_list = []
        for value in self.input_wordlist:
            word_list = get_words_to_katakana(value)
            hira_word = kata_to_hira(word_list)
            for value in list(hira_word[0]):
                self.hira_word_list.append(value)


class LstmNet():
    def __init__(self,input_len):
        self.input_len = input_len
        self.length_of_sequences = input_len
        self.in_out_neurons = 1
        self.hidden_neurons = 300
        self.out_nerouns = 500
        self.out_nerouns2 = input_len

        self.X_train = []
        self.Y_train = []

    def make_train_data(self,wordmap,input_wordlist):
        for i in range(len(input_wordlist)-5):
            if (len(input_wordlist)>5):
                self.X_train.append(np.r_[wordmap[input_wordlist[i]] ,wordmap[input_wordlist[i+1]],  wordmap[input_wordlist[i+2]], wordmap[input_wordlist[i+3]], wordmap[input_wordlist[i+4]] ])
                self.Y_train.append(wordmap[input_wordlist[i+5]])

        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)


        # self.X_train = self.X_train.reshape(len(self.X_train),1,len(self.X_train[0]))
        # self.Y_train = self.Y_train.reshape(len(self.Y_train),len(self.Y_train[0]))

        print(self.X_train.shape)
        self.X_train = self.X_train.reshape(len(self.X_train),1,len(self.X_train[0]))
        self.Y_train = self.Y_train.reshape(len(self.Y_train),len(self.Y_train[0]))
        print(self.X_train.shape)

    def make_net(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_neurons, input_shape=(1, len(self.X_train[0][0]))))
        # self.model.add(LSTM(self.hidden_neurons, input_shape=(1, 97)))
        # self.model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), return_sequences=False))
        self.model.add(Dense(self.length_of_sequences))
        self.model.add(Activation("softmax"))
        # self.model.add(Dense(self.out_nerouns))
        # self.model.add(Dense(self.out_nerouns2))
        # self.model.add(Activation("softmax"))

        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        loss='categorical_crossentropy'
        optimizer = "adam"
        optimizer = RMSprop(lr=0.01)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

    def train(self):
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=380, nb_epoch=250)
        # self.history = self.model.fit(self.X_train, self.Y_train, batch_size=400, nb_epoch=1, validation_split=0.05)


    def score(self):
        score = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        # print(score)
        print('test loss:', score[0])
        print('test acc:', score[1])


    def predict(self,st1,st2,st3,st4,st5,wordmap):
        # sp = wordmap[st]
        # sp = np.array([sp])
        # sp = np.array([sp])

        # wordlist = list("sあぁいぃうゔぅえぇおぉかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづってでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゃゆゅよょらりるれろわゐゑをんー?!()「」、。,.・e")
        # sp1 = wordmap["s"]
        # sp2 = wordmap[random.choice(wordlist)]
        # sp3 = wordmap[random.choice(wordlist)]

        sp = []
        sp.append(np.r_[st1, st2, st3,st4,st5])
        sp = np.array([sp])

        sp = sp.reshape(1,1,len(self.X_train[0][0]))
        # self.X_train.append(np.r_[wordmap[input_wordlist[i]] ,wordmap[input_wordlist[i+1]],  wordmap[input_wordlist[i+2]] ])
        # sp = sp.reshape(1,1,len(self.X_train[0][0]))

        predict_list = self.model.predict_on_batch(sp)
        # print(predict_list)

        # x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])
        # tlist = np.zeros(len(wordmap))
        # tlist[x] = 1
        # tlist = np.array(tlist)
        return predict_list

    def wait_controller(self,flag):
        try:
            if flag == "s":
                self.model.save_weights('./wait/param.hdf5')
            if flag == "l":
                self.model.load_weights('./wait/param.hdf5')
        except :
            print("no such file")


    def plot_history(self):
        # 精度の履歴をプロット
        # plt.plot(self.history.history['acc'],"o-",label="accuracy")
        # plt.plot(self.history.history['val_acc'],"o-",label="val_acc")
        # plt.title('model accuracy')
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy')
        # plt.legend(loc="lower right")
        # plt.show()

        # 損失の履歴をプロット
        plt.plot(self.history.history['loss'],"o-",label="loss",)
        # plt.plot(self.history.history['val_loss'],"o-",label="val_loss")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        plt.show()


def make_sentens(mynet,mydata):
    sentens = ""

    wordlist = list("sあぁいぃうゔぅえぇおぉかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづってでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゃゆゅよょらりるれろわゐゑをんー?!()「」、。,.・e")
    wlist1 = mydata.wordmap["s"]
    wlist2 = mydata.wordmap[random.choice(wordlist)]
    wlist3 = mydata.wordmap[random.choice(wordlist)]
    wlist4 = mydata.wordmap[random.choice(wordlist)]
    wlist5 = mydata.wordmap[random.choice(wordlist)]

    while(True):
        predict_list = mynet.predict(wlist1,wlist2,wlist3,wlist4,wlist5,mydata.wordmap)
        x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])

        # make list
        outlist = np.zeros(len(mydata.wordmap))
        outlist[x] = 1
        outlist = np.array(outlist)

        # list to word
        for value in mydata.wordmap.items() :
            if (np.dot(value[1],outlist) == 1):
            # if(np.sum(value[1] - outlist) == 0 ):
                    outstr = value[0]
                    sentens+=outstr

        wlist1 = wlist2
        wlist2 = wlist3
        wlist3 = wlist4
        wlist4 = wlist5
        wlist5 = outlist
        if(sentens[-1] == "e") :  break
        # if((sentens[-1] == "e") or (sentens[-1] == ".") or (sentens[-1] == "。")) :  break

    print(sentens)
    # print(tlist)
    # print(mydata.wordmap.values().index(tlist))
    # print(mydata.wordmap.keys()[mydata.wordmap.values().index(tlist)])



def main():
    flag = "m"
    mydata = DataSet()
    mydata.setdata()
    wordlist_len = mydata.get_wordlist_len()

    myfile = ReadFile()
    fdata = myfile.readfile("./text/kusanagi_notbof.txt")
    myfile.make_wordlist(fdata)

    myfile.to_hiragana()
    # print(myfile.hira_word_list)

    mynet = LstmNet(wordlist_len)
    mynet.make_train_data(mydata.wordmap, myfile.hira_word_list)
    mynet.make_net()


    # 学習
    if (flag == "l") :
        mynet.train()
        mynet.wait_controller("s")

    # modelに学習させた時の変化の様子をplot
    # mynet.plot_history()

    if (flag == "t") :
        mynet.wait_controller("l")
        mynet.predict("あ",mydata.wordmap)
        mynet.predict("か",mydata.wordmap)
        mynet.predict("さ",mydata.wordmap)


    # 文章生成
    if (flag == "m") :
        mynet.wait_controller("l")
        for i in range(10):
            make_sentens(mynet,mydata)

if __name__ == "__main__" :
    main()
