import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 输入数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender_data = np.array(['M', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M'])

# 将性别转换为数值（例如，'M' 可以表示为 1，'F' 可以表示为 0）
gender_data_numeric = np.where(gender_data == 'M', 1, 0)

# 创建特征矩阵
X = np.column_stack((height_data, weight_data))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, gender_data_numeric, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy * 100:.2f}%")

# 绘制数据和决策边界
plt.scatter(X[:, 0], X[:, 1], c=gender_data_numeric, cmap='viridis', edgecolors='k', marker='o')
plt.xlabel('身高')
plt.ylabel('体重')

# 绘制决策边界
coef = model.coef_[0]
intercept = model.intercept_[0]
line = lambda x: (-intercept - coef[0] * x) / coef[1]
x_vals = np.linspace(min(height_data), max(height_data), 100)
plt.plot(x_vals, line(x_vals), color='red', linestyle='dashed', label='Decision Boundary')

plt.legend()
plt.show()
