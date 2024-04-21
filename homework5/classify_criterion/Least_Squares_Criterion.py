import numpy as np

def LMSE(X, X_inverse, b, c,maxepoch,test_data):
    # 开始迭代
    i = 0
    e_res = []
    acc_res = []
    while (i<maxepoch):
        # 循环主体
        W = np.dot(X_inverse, b)
        print("epoch:",i+1)
        e = np.dot(X,W) - b
        e_res.append(np.sum(np.abs(e)))
        print("误差向量：", e)
        if np.all(e<0):
            print("无解")
            return np.zeros(81)
        if np.all(e <= 0.001):
            print("训练结束")
            return W
        #print("X*W-b=\n",X,"*",W,"-",b,"=",e)
        deta_b = c * (e + np.abs(e))
        b = b + deta_b  # 更新b
        print("更新b:")
        W = W + np.dot(X_inverse,deta_b)  # 更新W
        print("更新W:")
        # 训练集上测试
        success = 0
        for x in X:
            if np.dot(x, W) > 0:
                success = success + 1
        accuracy = float(success / 80)
        print('训练准确率：', accuracy)
        # 在测试集上测试
        success = 0
        for x in test_data:
            if np.dot(x, W) > 0:
                success = success + 1
        accuracy = float(success / 20)
        acc_res.append(accuracy)
        print('测试准确率：', accuracy)
        i = i+1
    return W,e_res,acc_res