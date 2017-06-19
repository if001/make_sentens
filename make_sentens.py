'''
lstmを使って文章生成

batch_input_shape=(None,\
                   LSTMの中間層に入力するデータの数（※文書データなら単語の数）,\
                   LSTM中間層に投入するデータの次元数（※文書データなら１次元配列なので1)
                  )
'''


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

import numpy as np
import numpy.random as rand

import sys

from mecab_test import get_words_hinshi
from mecab_test import kata_to_hira

class DataSet():
    def __init__(self):
        self.wordmap = {}

    def setdata(self):
        wordlist = list("sあぁいぃうゔぅえぇおぉかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづってでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゃゆゅよょらりるれろわゐゑをんー?!()「」、。,.・e")

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
            word_list = get_words_hinshi(value)
            hira_word = kata_to_hira(word_list)
            for value in list(hira_word[0]):
                self.hira_word_list.append(value)


class LstmNet():
    def __init__(self,input_len):
        self.input_len = input_len
        self.length_of_sequences = input_len
        self.in_out_neurons = 1
        self.hidden_neurons = 600
        self.out_nerouns = 500
        self.out_nerouns2 = input_len

        self.X_train = []
        self.Y_train = []

    def make_train_data(self,wordmap,input_wordlist):
        for i in range(len(input_wordlist)-1):
            # print(input_wordlist[i])
            # print(wordmap[input_wordlist[i]])
            self.X_train.append(wordmap[input_wordlist[i]])
            self.Y_train.append(wordmap[input_wordlist[i+1]])


        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)

        self.X_train = self.X_train.reshape(len(self.X_train),len(self.X_train[0]),1)
        self.Y_train = self.Y_train.reshape(len(self.X_train),len(self.X_train[0]))


    def make_net(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), return_sequences=False))
        # self.model.add(Dense(self.hidden_neurons))
        # self.model.add(Activation("relu"))
        self.model.add(Dense(self.out_nerouns))
        self.model.add(Dense(self.out_nerouns2))
        self.model.add(Activation("softmax"))

        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        optimizer = "adam"

        self.model.compile(loss=loss, optimizer=optimizer)
        # self.model.summary()

    def train(self):
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=400, nb_epoch=3, validation_split=0.05)

    def score(self):
        score = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        print(score)
        # print('test loss:', score[0])
        # print('test acc:', score[1])


    def predict(self,st,wordmap):
        sp = wordmap[st]
        sp = np.array([sp])
        sp = np.array([sp])
        sp = sp.reshape(1,len(self.X_train[0]),1)
        # print(sp.shape)
        predict_list = self.model.predict_on_batch(sp)
        print(predict_list)


        x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])
        tlist = np.zeros(len(wordmap))
        tlist[x] = 1
        tlist = np.array(tlist)




    def wait_controller(self,flag):
        try:
            if flag == "s":
                self.model.save_weights('./wait/param.hdf5')
            if flag == "l":
                self.model.load_weights('./wait/param.hdf5')
        except :
            print("no such file")



def make_sentens(mynet,mydata):
    word = "s"
    sentens = []
    while(True):
        # print(word)
        predict_list = mynet.predict(word,mydata.wordmap)
        # print(predict_list)
        x = rand.choice(len(predict_list[0]), 1, p=predict_list[0])
        tlist = np.zeros(len(mydata.wordmap))
        tlist[x] = 1
        tlist = np.array(tlist)

        for value in mydata.wordmap.items() :
            # print(value[1],tlist,value[0])
            if(np.sum(value[1] - tlist) == 0 ): word = value[0]
            sentens.append(word)
        if((sentens[-1] == "e") or (sentens[-1] == ".") or (sentens[-1] == "。")) :  break

    print(sentens)

    # print(tlist)
    # print(mydata.wordmap.values().index(tlist))
    # print(mydata.wordmap.keys()[mydata.wordmap.values().index(tlist)])




def main():
    flag = "l"
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

    # テスト
    if (flag == "t") :
        mynet.wait_controller("l")
        mynet.predict("あ",mydata.wordmap)
        mynet.predict("い",mydata.wordmap)
        mynet.predict("う",mydata.wordmap)

    # 文章生成
    if (flag == "m") :
        mynet.wait_controller("l")
        make_sentens(mynet,mydata)


if __name__ == "__main__" :
    main()
