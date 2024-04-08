import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import svm

# 身高、体重、性别数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender_data = np.array(['M', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M'])

# 绘制性别在对应身高体重平面空间中的分布
colors = {'M': 'lightblue', 'F': 'pink'}
plt.scatter(height_data, weight_data, c=[colors[gender] for gender in gender_data], label=gender_data)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Gender in Height-Weight Space')
plt.legend()

# 询问用户选择核函数
kernel_choice = input("请选择核函数（1: 线性核函数, 2: 多项式核函数, 3: 高斯核函数）: ")

# 使用支持向量机进行训练
if kernel_choice == '1':
    clf = svm.SVC(kernel='linear')
elif kernel_choice == '2':
    degree = int(input("请输入多项式次数（2, 3, 4等）: "))
    clf = svm.SVC(kernel='poly', degree=degree)
elif kernel_choice == '3':
    gamma = float(input("请输入高斯核函数的参数 gamma（推荐0.1）: "))
    clf = svm.SVC(kernel='rbf', gamma=gamma)
else:
    print("无效选择。")

# 训练支持向量机模型
clf.fit(np.column_stack((height_data, weight_data)), gender_data)

# 输出分界面函数方程
if kernel_choice == '1':
    print("线性核函数分类器方程: " + str(clf.coef_[0][0]) + " * Height + " + str(clf.coef_[0][1]) + " * Weight + " + str(clf.intercept_[0]) + " = 0")
elif kernel_choice == '2':
    for i in range(degree + 1):
        print(f"多项式核函数（次数 {i}）分类器方程: {clf.dual_coef_[0][i]} * (Height ** {i}) + {clf.dual_coef_[0][i + degree + 1]} * (Weight ** {i}) + {clf.intercept_[0]} = 0")
elif kernel_choice == '3':
    print(f"高斯核函数分类器方程: 未显示，因为是在高维空间")

# 绘制分界面结果
plt.scatter(height_data, weight_data, c=[colors[gender] for gender in gender_data], label=gender_data)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 生成网格点，用于绘制分界面
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# 将结果放回图中
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot with SVM Decision Boundary')
plt.legend()
plt.show()
