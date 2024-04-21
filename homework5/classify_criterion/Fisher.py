import numpy as np
def fisher(data_set):
    # 求均值
    X1 = []
    X2 = []
    X1_sum = np.zeros(len(data_set[0][0]))
    num1 = 0
    num2 = 0
    X2_sum = X1_sum
    for x, y in data_set:
        if y == -1:
            X1_sum = X1_sum + x
            num1 = num1 + 1
            X1.append(x)
        else:
            X2_sum = X2_sum + x
            num2 = num2 + 1
            X2.append(x)
    X1_mean = X1_sum / num1
    X2_mean = X2_sum / num2
    # 计算类内散度矩阵
    S1 = np.zeros(len(data_set[0][0]))
    S2 = S1
    for x in X1:
        S1 = S1 + np.dot((x - X1_mean),np.transpose(x - X1_mean))
    for x in X2:
        S2 = S2 + np.dot((x - X2_mean),np.transpose(x - X2_mean))

    S = S1 + S2
    # 计算最佳投影方向
    W = S*(X1_mean - X2_mean)

    return W, X1 ,X2