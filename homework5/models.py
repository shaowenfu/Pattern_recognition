from classify_criterion.perceptron import perceptron
import numpy as np
import matplotlib.pyplot as plt
from utils import inverse_matrix,Weight
from classify_criterion.Least_Squares_Criterion import LMSE
from classify_criterion.Fisher import fisher
from classify_criterion.Multi_cla import Multi_train
import matplotlib
matplotlib.use('TkAgg')  # 选择一个合适的后端TkAgg绘制图表

class Perceptron:
    def __init__(self,test_label,test_set,train_label,train_set,data_dim):
        self.test_label = test_label
        self.test_set = test_set
        self.train_label = train_label
        self.train_set = train_set
        self.data_dim = data_dim
        self.w = np.zeros(self.data_dim)
    def fuc(self):
        # 选取标签为0金鱼，标签为1青蛙的训练和测试集
        i = 0
        test_data = []  # 测试集
        for value in self.test_label:
            if value == 0 or value == 1:
                if value == 0:
                    value = -1  # 金鱼标签设为-1
                test_data.append([self.test_set[i], value])
            i = i + 1
        train_data = []  # 训练集
        i = 0
        for label in self.train_label:
            if label == 0 or label == 1:
                if label == 0:
                    label = -1  # 金鱼标签设为-1
                train_data.append([self.train_set[i], label])
            i = i + 1
        # 开始训练
        w0 = np.zeros(self.data_dim)
        b0 = 0
        alpha = 0.001
        maxepoch = 10000
        self.w, b = perceptron(train_data, w0, b0, alpha, maxepoch)
        # 在测试集上测试
        success = 0
        for x, y in test_data:
            if y * (np.dot(x, self.w)) + b > 0:
                success = success + 1
        accuracy = float(success / 20)
        print('测试准确率：', accuracy)

class LSC:
    def __init__(self,test_label,test_set,train_label,train_set,data_dim):
        self.test_label = test_label
        self.test_set = test_set
        self.train_label = train_label
        self.train_set = train_set
        self.data_dim = data_dim
    def fuc(self):
        # 选取标签为0金鱼，标签为1青蛙的训练和测试集
        i = 0
        test_data = []  # 测试集
        for value in self.test_label:
            if value == 0 or value == 1:
                list = []
                list = np.append(self.test_set[i], 1)
                if value == 0:
                    value = -1  # 金鱼标签设为-1
                    # 使用列表推导式将列表中的每个元素乘以 -1
                    list = np.array([-x for x in list])
                test_data.append(list)
            i = i + 1
        train_data = []  # 训练集
        i = 0
        for label in self.train_label:
            if label == 0 or label == 1:
                list = []
                list = np.append(self.train_set[i], 1)
                if label == 0:
                    label = -1  # 金鱼标签设为-1
                    # 使用列表推导式将列表中的每个元素乘以 -1
                    list = np.array([-x for x in list])
                train_data.append(list)
            i = i + 1
        X = train_data
        X_inverse = inverse_matrix(X)  # 求伪逆矩阵
        # 初始化参数
        b = np.ones(80)
        c = 0.01
        maxepoch = 67
        train_sumples = 80
        test_sumples = 20
        W_res,e_res,acc_res = LMSE(X, X_inverse, b, c,maxepoch,test_data)
        if np.all(W_res == 0):
            print("线性不可分")
        print("权重结果：",W_res)
        # 可视化

        # x 轴数据，假设为迭代次数
        iterations = range(1, len(e_res) + 1)

        # 创建新的图像
        plt.figure(figsize=(10, 5))

        # 绘制误差折线图
        plt.subplot(1, 2, 1)  # 子图1
        plt.plot(iterations, e_res, marker='o', color='b', label='Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Error vs Iterations')
        plt.legend()

        # 绘制准确率折线图
        plt.subplot(1, 2, 2)  # 子图2
        plt.plot(iterations, acc_res, marker='o', color='r', label='Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.legend()

        # 显示图像
        plt.tight_layout()
        plt.show()

class Fisher:
    def __init__(self, test_label, test_set, train_label, train_set, data_dim):
        self.test_label = test_label
        self.test_set = test_set
        self.train_label = train_label
        self.train_set = train_set
        self.data_dim = data_dim
    def fuc(self):
        # 选取标签为0金鱼，标签为1青蛙的训练和测试集
        i = 0
        test_data = []  # 测试集
        for value in self.test_label:
            if value == 0 or value == 1:
                if value == 0:
                    value = -1  # 金鱼标签设为-1
                test_data.append([self.test_set[i], value])
            i = i + 1
        train_data = []  # 训练集
        i = 0
        for label in self.train_label:
            if label == 0 or label == 1:
                if label == 0:
                    label = -1  # 金鱼标签设为-1
                train_data.append([self.train_set[i], label])
            i = i + 1

        # 计算最佳投影向量
        W, X1, X2 = fisher(train_data)

        # 计算投影
        # 将每个列表元素与 W 进行矩阵乘法
        X1_project = [np.dot(W, x) for x in X1]
        X2_project = [np.dot(W, x) for x in X2]
        # 计算最佳分界点
        median_class1 = np.median(X1_project)
        median_class2 = np.median(X2_project)
        # 计算中位数的平均值作为分界点
        if median_class1 > median_class2:
            sign = 1
        else:
            sign = -1
        threshold = (median_class1 + median_class2) / 2

        # 测试分类准确度
        success = 0
        for x,y in test_data:
            x_project = np.dot(W,x)
            if (x_project - threshold)*sign >0:
                if y == -1:
                    success = success + 1
            else:
                if y == 1:
                    success = success + 1
        print("测试集上分类准确性：",success/20)

class Multi_cla:
    def __init__(self,test_label, test_set, train_label, train_set, data_dim):
        self.test_label = test_label
        self.test_set = test_set
        self.train_label = train_label
        self.train_set = train_set
        self.data_dim = data_dim

    def fuc(self):
        W = Weight(left_index=0,right_index=9,data=np.zeros(self.data_dim))
        left = 0
        right = 9
        success = 0
        Multi_train(W,left,right,self.test_label, self.test_set, self.train_label, self.train_set, self.data_dim)
        m = 0
        for x in self.test_set:
            y = self.test_label[m]
            print(f"开始对测试集第{m+1}个数据{y}分类")
            i = 0
            w_temp = W
            fail = 0
            while(w_temp != None and w_temp.left_index != w_temp.right_index and fail == 0):
                flag = int((w_temp.left_index+w_temp.right_index)/2)
                if y<= flag:
                    v = -1
                else:
                    v = 1
                if (np.dot(w_temp.data,x)+w_temp.b)*v <=0:
                    if v == -1:
                        print(f"分类失败，将{y}错分到{flag+1}和{w_temp.right_index}之间")
                    else:
                        print(f"分类失败，将{y}错分到{w_temp.left_index}和{flag}之间")
                    fail = 1
                elif v ==-1:
                    print(f"向{w_temp.left_index}和{flag}方向搜索")
                    w_temp = w_temp.left
                else:
                    print(f"向{flag+1}和{w_temp.right_index}方向搜索")
                    w_temp = w_temp.right
            if fail == 0:
                success = success + 1
            m = m + 1
        print("测试集准确率：",success/100)