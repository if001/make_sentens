d = {"a":1,"b":2}

for value in d.items() :
    if (value[1] == 1 ): key = value[0]
    print(value)

print(key)


import numpy as np
a = np.array([1,1,1])
b = np.array([2,2,2])

print(np.sum(b-a))



import numpy.random as rd
p = np.array([0.1, 0.1, 0.1, 0.7])
x2=rd.choice(len(p), 1, p=p)
print(x2)
