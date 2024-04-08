# 假设我们有一个人口普查系统，我们希望根据人的年龄和收入水平来预测其是否会购买某种产品。我们将两个类别定义为：
# - 类别 *ω*1: 不购买产品的人群
# - 类别 *ω*2: 购买产品的人群
# 我们将人的特征定义为：
# - 特征 *x*1: 年龄（以年为单位）
# - 特征 *x*2: 收入水平（以年收入为单位）
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(0)

# 定义类别的先验概率
p_w1 = 0.3  # 类别1的先验概率
p_w2 = 1 - p_w1  # 类别2的先验概率

# 定义类别1的年龄和收入的均值和协方差
mean_w1 = [35, 50000]
cov_w1 = [[100, 10000], [10000, 1000000]]

# 定义类别2的年龄和收入的均值和协方差
mean_w2 = [45, 70000]
cov_w2 = [[100, 20000], [20000, 1500000]]

# 生成示例数据
n_samples = 100
X_w1 = np.random.multivariate_normal(mean_w1, cov_w1, n_samples)
X_w2 = np.random.multivariate_normal(mean_w2, cov_w2, n_samples)

# 将类别标签添加到数据中
y_w1 = np.ones((n_samples, 1))  # 类别1标签
y_w2 = np.ones((n_samples, 1)) * 2  # 类别2标签

# 合并数据和标签
data_w1 = np.hstack((X_w1, y_w1))
data_w2 = np.hstack((X_w2, y_w2))
data = np.vstack((data_w1, data_w2))

# 打乱数据顺序
np.random.shuffle(data)

# 将数据划分为特征和标签
X = data[:, :-1]
y = data[:, -1].astype(int)

# 查看生成的数据
print("示例数据：")
print(X[:5])
print(y[:5])
