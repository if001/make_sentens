import numpy as np
arr1 = [1,2,3]
arr1 = np.array(arr1)

arr2 = ["a","b","c"]
arr2 = np.array(arr2)

# print(arr1+arr2)
# print(np.r_[arr1, arr2])




# from keras.utils.data_utils import get_file
# import numpy as np


# text = open('./text/tmp.txt').read().lower()
# print(text)
# print('corpus length:', len(text))

# chars = sorted(list(set(text)))
# print('total chars:', len(chars))
# print(len(chars))

# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# print(char_indices)
# print(indices_char)

# # cut the text in semi-redundant sequences of maxlen characters
# maxlen = 3
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])
# print('nb sequences:', len(sentences))


# print(sentences)
# print(next_chars)

# print('Vectorization...')
# X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         X[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

# print(X.shape)
# print(maxlen,len(chars))
# print(y.shape)
