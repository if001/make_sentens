

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

import numpy as np

class DataSet():
    def __init__(self):
        self.train = []

    def setdata(self):
        wordlist = ["あ","い","う","え","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ","ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","つ","づ","て","で","と","ど","な","に","ぬ","ね","の","は","ば","ひ","び","ふ","ぶ","へ","べ","ほ","ぼ","ま","み","む","め","も","や","ゆ","よ","わ","を","ん","、","。",",","."]

        for value in wordlist:
            tlist = np.zeros(len(wordlist))
            tlist[wordlist.index(value)] = 1
            self.train.append(tlist)

        self.train = np.array(self.train)


class ReadFile():
    def __init__(self):
        self.input_wordlist = []

    def readfile(self,fname):
        try:
            return open(fname)
        except IOError:
            print('cannot be opened.')

        for word in f.read():
            if word != "\n" :
                self.intput_wordlist.append(word)


class LstmNet():
    def __init__(self): pass

    def make_net(self):
        self.model = Sequential()
        self.model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
        self.model.add(Dense(in_out_neurons))
        self.model.add(Activation("linear"))
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop")
        self.model.fit(X_train, y_train, batch_size=600, nb_epoch=15, validation_split=0.05)


def main():
    mydata = DataSet()
    mydata.setdata()

    myfile = ReadFile()
    myfile.readfile("test.txt")



    
if __name__ == "__main__" :
    main()
