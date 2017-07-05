'''
lstmを使って文章生成
江戸川乱歩モデル
word2vecを自作して、vector化
...しようとしたけど、辞書作ってるだけでvector化はまだ使ってない


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

from mecab_test import get_words

import myword2vec
#from myword2vec import 

#import matplotlib.pyplot as plt
import pylab as plt


from tqdm import tqdm #プログレスバー

class DataSet():
    def __init__(self):pass

    def setdata(self,flag):
        self.myword2vecTrainData = myword2vec.TrainData()
        #self.myword2vecTrainData.readfile("./aozora_text/re_re_test.txt")
        self.myword2vecTrainData.readfile("./aozora_text/re_re_akumano_monsho.txt")
        #mydata.readfile("./aozora_text/files_all.txt")
        self.myword2vecTrainData.make_dict()

        # self.myword2vecNet = myword2vec.Net()

        # self.myword2vecNet.neuron_num_set(self.myword2vecTrainData)
        # self.myword2vecNet.make_net()

        # if flag == "l" :
        #     self.myword2vecNet.make_one_hot(self.myword2vecTrainData)
        #     self.myword2vecNet.train_net()
        #     self.myword2vecNet.wait_controller("s")

        # if flag == "m":
        #     self.myword2vecNet.wait_controller("l")


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
                self.input_wordlist.append(get_words(word))




class LstmNet():
    def __init__(self):
        self.hidden_neurons = 1200

        self.X_train = []
        self.Y_train = []
        self.gram = 4


    def input_len_set(self,mydata):
        self.input_len = len(mydata.myword2vecTrainData.worddict)*self.gram
        self.output_len = len(mydata.myword2vecTrainData.worddict)

    def make_train_data(self,mydata,input_wordlist):
        print("make_train_data")
        for i in tqdm(range(len(input_wordlist))):
            for j in range(len(input_wordlist[i])-(self.gram+1)):
                tmp_train_data1 = np.zeros(len(mydata.myword2vecTrainData.worddict))
                tmp_train_data1[np.where(mydata.myword2vecTrainData.worddict == input_wordlist[i][j])] = 1
                tmp_train_data2 = np.zeros(len(mydata.myword2vecTrainData.worddict))
                tmp_train_data2[np.where(mydata.myword2vecTrainData.worddict == input_wordlist[i][j+1])] = 1
                tmp_train_data3 = np.zeros(len(mydata.myword2vecTrainData.worddict))
                tmp_train_data3[np.where(mydata.myword2vecTrainData.worddict == input_wordlist[i][j+2])] = 1
                tmp_train_data4 = np.zeros(len(mydata.myword2vecTrainData.worddict))
                tmp_train_data4[np.where(mydata.myword2vecTrainData.worddict == input_wordlist[i][j+3])] = 1

                tmp_train_data5 = np.zeros(len(mydata.myword2vecTrainData.worddict))
                tmp_train_data5[np.where(mydata.myword2vecTrainData.worddict == input_wordlist[i][j+4])] = 1


                self.X_train.append(np.r_[tmp_train_data1,tmp_train_data2,tmp_train_data3,tmp_train_data4])
                self.Y_train.append(tmp_train_data5)


        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        print(self.X_train.shape)
        print(self.Y_train.shape)


        print(self.X_train.shape)
        self.X_train = self.X_train.reshape(len(self.X_train), 1, len(self.X_train[0]))
        print(self.X_train.shape)
        # sys.exit(0)


    def make_net(self):
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_neurons, input_shape=(1, self.input_len)))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation("softmax"))

        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        loss='categorical_crossentropy'
        optimizer = "adam"
        optimizer = RMSprop(lr=0.01)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

    def train(self):
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=500, nb_epoch=50)
        # self.history = self.model.fit(self.X_train, self.Y_train, batch_size=400, nb_epoch=1, validation_split=0.05)


    def score(self):
        score = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        # print(score)
        print('test loss:', score[0])
        print('test acc:', score[1])


    def predict(self,st1,st2,st3):
        sp = []
        sp.append(np.r_[st1, st2, st3])
        sp = np.array([sp])
        sp = sp.reshape(1,1,self.input_len)

        predict_list = self.model.predict_on_batch(sp)
        # print(predict_list)

        # x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])
        # tlist = np.zeros(len(wordmap))
        # tlist[x] = 1
        # tlist = np.array(tlist)
        return predict_list

    def wait_controller(self,flag):
        try:
            if flag == "save":
                self.model.save_weights('./wait/param_make_sentens_wordvec.hdf5')
            if flag == "load":
                self.model.load_weights('./wait/param_make_sentens_wordvec.hdf5')
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
    #sentens = []
    sentens = ""

    st1 = random.choice(mydata.myword2vecTrainData.worddict)
    wbin1 = np.zeros(len(mydata.myword2vecTrainData.worddict))
    wbin1[np.where(mydata.myword2vecTrainData.worddict == st1)] = 1

    st2 = random.choice(mydata.myword2vecTrainData.worddict)
    wbin2 = np.zeros(len(mydata.myword2vecTrainData.worddict))
    wbin2[np.where(mydata.myword2vecTrainData.worddict == st2)] = 1

    st3 = random.choice(mydata.myword2vecTrainData.worddict)
    wbin3 = np.zeros(len(mydata.myword2vecTrainData.worddict))
    wbin3[np.where(mydata.myword2vecTrainData.worddict == st3)] = 1


    while(True):
        predict_list = mynet.predict(wbin1,wbin2,wbin3)
        x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])
        outstr = mydata.myword2vecTrainData.worddict[x]
        #sentens.append(outstr)
        sentens += outstr[0]
        print(outstr)

        # make binary
        outbin = np.zeros(len(mydata.myword2vecTrainData.worddict))
        outbin[x] = 1
        outbin = np.array(outbin)

        wbin1 = wbin2
        wbin2 = wbin3
        wbin3 = outbin

        if((sentens[-1] == "e") or (sentens[-1] == ".") or (sentens[-1] == "。")) :  break
        if len(sentens) > 100: break

    print(sentens)


def main():
    flag = "learn"
    mydata = DataSet()
    mydata.setdata("m")


    myfile = ReadFile()
    #fdata = myfile.readfile("./aozora_text/re_re_test.txt")
    fdata = myfile.readfile("./aozora_text/re_re_akumano_monsho.txt")
    myfile.make_wordlist(fdata)

    mynet = LstmNet()
    mynet.input_len_set(mydata)
    mynet.make_net()

    # 学習
    if (flag == "learn") :
        mynet.make_train_data(mydata,myfile.input_wordlist)
        mynet.train()
        mynet.wait_controller("save")

    # # modelに学習させた時の変化の様子をplot
    # # mynet.plot_history()

    # if (flag == "t") :
    #     mynet.wait_controller("l")
    #     mynet.predict("あ",mydata.wordmap)
    #     mynet.predict("か",mydata.wordmap)
    #     mynet.predict("さ",mydata.wordmap)


    # 文章生成
    if (flag == "make") :
        mynet.wait_controller("load")
        make_sentens(mynet,mydata)
        make_sentens(mynet,mydata)
        make_sentens(mynet,mydata)
        make_sentens(mynet,mydata)
    #     make_sentens(mynet,mydata)
    #     make_sentens(mynet,mydata)
    #     make_sentens(mynet,mydata)
    #     make_sentens(mynet,mydata)

if __name__ == "__main__" :
    main()
