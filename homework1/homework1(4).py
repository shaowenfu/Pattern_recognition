import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 已知数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])  # 1 for Male, 0 for Female

# 训练一个支持向量机分类器
X = np.column_stack((height_data, weight_data))
clf = SVC(kernel='linear')
clf.fit(X, gender)

# 绘制训练样本的散点图
plt.scatter(X[:, 0], X[:, 1], c=np.where(gender == 1, 'b', 'pink'), edgecolors='k', marker='o')
plt.xlabel('Height')
plt.ylabel('Weight')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# 将结果放回图形
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# 显示图形
plt.title('Decision Boundary and Training Samples')
plt.show()
