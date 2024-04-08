import numpy as np
def perceptron(data_set,w,b,alpha = 0.5,maxepoch = 100):
    i =0
    flag = True
    while(flag and i<=maxepoch):
        print(f"epoch:{i+1}")
        i = i + 1
        flag = 0
        success = 80
        for x, y in data_set:
            res = y * np.dot(x, w) + b
            if res <= 0:
                # 更新w0和b0
                w = w + alpha * y * x
                b = b + alpha * y
                flag = 1
                success = success -1
        print(f"accuracy:{success}")
    return w,b