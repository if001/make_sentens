
"""
学習データと教師データ作るのがおもすぎたのでcythonにまかせる

"""


#numpyをcythonから扱えるようにするためにcimportを使用します
import numpy as np
cimport numpy as np


def make_train_data(input_len,worddict,wordlists,window):
    print("cython!! make one hot vector")
    print("wordlist length :",len(wordlists))

    train_x =[]
    train_y =[]
    cdef int i = 0
    cdef int j = 0
    cdef int ls,le
    cdef int wordlists_len = len(wordlists)
    cdef int worddict_len = len(worddict)

    # cdef double[:] tmp_train_x_ = np.zeros(input_len, dtype=np.float64)
    # cdef double[:] tmp_train_y = np.zeros(0, dtype=np.float64)

    for i in range(wordlists_len):
        tmp_train_x = np.zeros(input_len)
        tmp_train_y = np.zeros(0)
        # cdef int[:] tmp_train_x_ = np.zeros(input_len, dtype=np.int32)
        # cdef int[:] tmp_train_y = np.zeros(0, dtype=np.int32)
        tmp_train_x_ = np.zeros(input_len, dtype=np.float64)
        tmp_train_y = np.zeros(0, dtype=np.float64)

        # for train x
        tmp_train_x[np.where(worddict == wordlists[i])] = 1
        train_x.append(tmp_train_x)

        # for train y
        if window == 0:
            ls = 0
            le = 1
        else:
            ls = -window
            le = window + 1

        for j in range(ls,le):
            tmp_tmp_train_y = np.zeros(worddict_len)
            if (( i + j ) > 0) and ((i + j) < len(wordlists)):
                tmp_tmp_train_y[np.where(worddict == wordlists[i+j])] = 1
            tmp_train_y.append(tmp_tmp_train_y)
            #tmp_train_y = np.append(tmp_train_y,tmp_tmp_train_y)

        train_y.append(tmp_train_y)

    return train_x,train_y


# def make_train_data(input_len,worddict,wordlists,window):
#     return play(input_len,worddict,wordlists,window)

