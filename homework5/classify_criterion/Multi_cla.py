from classify_criterion.perceptron import perceptron
from utils import Weight
import numpy as np
def Multi_train(W,left,right, test_label, test_set, train_label, train_set, data_dim):
    print(f"开始分类{left}到{right}")
    w0 = np.zeros(data_dim)
    b0 = 0
    alpha = 0.01
    maxepoch = 100
    # 如果已经只剩一类，分类完成
    if left == right:
        W.left_index = left
        W.right_index = right
        print("分类完成")
        return
    elif right - left == 1:
        # 数据准备
        train_data = []
        j = 0
        for label in train_label:
            if label == left:
                value = -1
                train_data.append([train_set[j],value])
            if label == right:
                value = 1
                train_data.append([train_set[j],value])
            j = j + 1
        test_data = []
        k=0
        for label in test_label:
            if label == left:
                value = -1
                test_data.append([test_set[k],value])
            if label == right:
                value = 1
                test_data.append([test_set[k],value])
            k = k + 1
        # 调用二分类模型分类
        w,b = perceptron(train_data, w0, b0, alpha, maxepoch)
        W.data = w
        W.b = b
        return
    else:
        flag = int((right+left)/2)
        # 数据准备
        train_data = []
        j = 0
        for label in train_label:
            if label == left:
                value = -1
                train_data.append([train_set[j],value])
            if label == right:
                value = 1
                train_data.append([train_set[j],value])
            j = j + 1
        test_data = []
        k=0
        for label in test_label:
            if label == left:
                value = -1
                test_data.append([test_set[k],value])
            if label == right:
                value = 1
                test_data.append([test_set[k],value])
            k = k + 1
        # 调用二分类模型分类
        w,b = perceptron(train_data, w0, b0, alpha, maxepoch)
        W.data = w
        W.b = b
        w1 = Weight(left_index=left, right_index=flag,data=np.zeros(data_dim))
        w2 = Weight(left_index=flag+1, right_index=right, data=np.zeros(data_dim))
        W.left = w1
        W.right = w2
        Multi_train(w1,  left, flag, test_label, test_set, train_label, train_set, data_dim)
        Multi_train(w2, flag+1, right, test_label, test_set, train_label, train_set, data_dim)


