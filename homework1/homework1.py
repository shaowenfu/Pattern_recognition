import numpy as np

def linear_regression(X, y):
    X = np.column_stack((np.ones(len(X)), X))
    # 计算回归系数
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return beta


def predict(beta, new_data):
    # 在新数据前添加一列全为 1 的列
    new_data = np.column_stack((np.ones(len(new_data)), new_data))

    # 进行预测
    predictions = np.dot(new_data, beta)

    return predictions


# 已知数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender_labels = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])  # M:1, F:0

# 特征矩阵
X = np.column_stack((height_data, weight_data))

# 训练回归模型
beta = linear_regression(X, gender_labels)
b0 = beta[0]
b1 = beta[1]
b2 = beta[2]
print()
print
# 新的身高体重数据
new_data = np.array([[140, 70], [150,60], [145, 65],[160,75]])

# 预测性别
predictions = predict(beta, new_data)

# 打印预测结果
for i in range(len(new_data)):
    predicted_gender = 'M' if predictions[i] >= 0.5 else 'F'
    print(f'预测身高 {new_data[i][0]}，体重 {new_data[i][1]} 的性别为: {predicted_gender}')
