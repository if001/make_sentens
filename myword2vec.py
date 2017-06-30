from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras import optimizers
from keras.layers import Input
from keras.models import Model

import numpy as np
import sys
import re


from tqdm import tqdm #プログレスバー



#import pyximport; pyximport.install()
#import makeTrainData

class TrainData():
    def __init__(self):
        self.wordlists = np.zeros(0)
        self.worddict = np.zeros(0)

    def readfile(self,fname):
        try:
            with open('./'+fname,'r') as file:
                for line in file:
                    line = re.sub(r'\n', "", line)
                    if len(line) != 0:
                        self.wordlists = np.append(self.wordlists,line.split(" "))
        except:
            print("not such file")
            sys.exit(0)

    def make_dict(self):
        self.worddict = np.append(self.worddict,"")
        for word in self.wordlists :
            if ( str(word) in self.worddict) == False :
                self.worddict = np.append(self.worddict,word)

        print("word dict length : ",len(self.worddict))

class Net():
    def __init__(self):
        self.window = 3
        self.BATCH_SIZE = 600
        self.STEP = 1

    def neuron_num_set(self,mydata):
        self.input_len = len(mydata.worddict)
        self.hidden_len = 200
        self.output_len = len(mydata.worddict)*(self.window*2+1)

    def make_one_hot(self,mydata):
        # self.train_x,self.train_y = makeTrainData.make_train_data(self.input_len,mydata.worddict
        #                                                           ,mydata.wordlists,self.window)
        # self.train_x = np.array(self.train_x)
        # self.train_y = np.array(self.train_y)

        self.train_x = []
        self.train_y = []

        print("make one hot vector")
        print("wordlist length :",len(mydata.wordlists))

        pbar = tqdm(total=len(mydata.wordlists)) #プログレスバー
        for i in range(len(mydata.wordlists)):
            pbar.update(i) #プログレスバー
            #pbar.update(1/len(mydata.wordlists)) #プログレスバー
            tmp_train_x = np.zeros(self.input_len)
            tmp_train_y = np.zeros(0)

            # for train x
            tmp_train_x[np.where(mydata.worddict == mydata.wordlists[i])[0][0]] = 1
            self.train_x.append(tmp_train_x)

            # for train y
            if self.window == 0:
                ls = 0
                le = 1
            else:
                ls = -self.window
                le = self.window+1

            for j in range(ls,le):
                #print(len(tmp_tmp_train_y))
                tmp_tmp_train_y = np.zeros(len(mydata.worddict))
                if (( i + j ) > 0) and ((i + j) < len(mydata.wordlists)):
                    tmp_tmp_train_y[np.where(mydata.worddict == mydata.wordlists[i+j])[0][0]] = 1
                tmp_train_y = np.append(tmp_train_y,tmp_tmp_train_y)
            self.train_y.append(tmp_train_y)

        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)



    def make_net(self):
        self.model = Sequential()

        self.model.add(Dense(self.hidden_len , input_shape=(self.input_len,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = "binary_crossentropy"
        loss='categorical_crossentropy'
        # モデルをコンパイル
        self.model.compile(loss=loss, optimizer=sgd)
        # # モデル表示
        # self.model.summary()


    def train_net(self):
        # モデルの訓練
        train = self.model.fit(self.train_x, self.train_y,
                               batch_size = self.BATCH_SIZE,
                               nb_epoch = self.STEP,
                               verbose=1,
                               validation_split=0.1)

    def vec_to_word(self):
        test = np.zeros(self.hidden_len)
        # test = [ 0 for i in range(self.hidden_len) ]
        # test[100] = 1
        # test = test.reshape(1,200)
        test = np.array([test])
        test = test.reshape(1,200)

        from keras import backend as K
        hidden_layer_output = K.function([self.model.layers[2].input],
                                          [self.model.layers[3].output])

        word_vec = hidden_layer_output([test])[0]
        print(word_vec)


    def predict(self,mydata,st):
        test_x = np.zeros(self.input_len)
        test_x[np.where(mydata.worddict == st)] = 1

        test_x = np.array([test_x])
        predict_list = self.model.predict_on_batch(test_x)
        print("predict :",st," : ",predict_list)

        #score = self.model.evaluate(mydata.test_images, mydata.test_labels, verbose=0)
        # print('test loss:', score[0])
        # print('test acc:', score[1])


    def wait_controller(self,flag):
        try:
            if flag == "s":
                self.model.save_weights('./wait/mywordvec_param.hdf5')
            if flag == "l":
                self.model.load_weights('./wait/mywordvec_param.hdf5')
        except :
            print("no such file")


def main():
    mydata = TrainData()
    #mydata.readfile("./aozora_text/re_re_test.txt")
    mydata.readfile("./aozora_text/re_re_akumano_monsho.txt")
    #mydata.readfile("./aozora_text/files_all.txt")
    mydata.make_dict()


    flag = "m" #学習orテストするかのフラグ
    mynet = Net()
    mynet.neuron_num_set(mydata)
    mynet.make_net()

    if flag == "l" :
        mynet.make_one_hot(mydata)
        mynet.train_net()
        mynet.wait_controller("s")

    if flag == "m" :
        mynet.predict(mydata,"私")
        mynet.wait_controller("l")

    if flag == "v" :
        mynet.vec_to_word()


if __name__ == "__main__":
    main()

