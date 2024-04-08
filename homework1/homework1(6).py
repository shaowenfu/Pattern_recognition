import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
# 身高、体重、性别数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender_data = np.array(['M', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M'])


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def plot_decision_boundary(height_data, weight_data, gender_data, degree, test_height, test_weight, test_gender):
    # 将性别转换为数字标签，M: 0, F: 1
    gender_labels = np.where(gender_data == 'M', 0, 1)

    # 绘制性别在对应身高体重平面空间中的分布
    plt.scatter(height_data[gender_labels == 0], weight_data[gender_labels == 0], c='lightblue', label='Male')
    plt.scatter(height_data[gender_labels == 1], weight_data[gender_labels == 1], c='pink', label='Female')

    # 使用多项式回归进行拟合
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(np.column_stack((height_data, weight_data)), gender_labels)

    # 绘制分界面
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放回散点图
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[0.5], linestyles=['-'], alpha=0.5)

    # 显示图例和图像
    plt.legend()
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Gender Distribution and Decision Boundary')

    # 输出分界面的函数表达式
    feature_names = PolynomialFeatures(degree).fit(np.column_stack((height_data, weight_data))).get_feature_names_out(['height', 'weight'])
    coefficients = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_

    terms = [f"{coeff:.2f} * {name}" for coeff, name in zip(coefficients, feature_names)]
    equation = f"Decision Boundary Function: {intercept:.2f} + {' + '.join(terms)}"
    print(equation)

    # 用训练得到的模型对新数据进行预测
    test_data = np.column_stack((test_height, test_weight))
    test_gender_labels = model.predict(test_data)

    # 将预测结果转换为性别标签
    predicted_gender = np.where(test_gender_labels == 0, 'M', 'F')

    # 输出预测结果和实际性别
    print("\nPredicted Gender:", predicted_gender)
    print("Actual Gender   :", test_gender)

    # 计算预测准确率
    accuracy = np.mean(test_gender == predicted_gender)
    print("\nAccuracy:", accuracy)

    plt.show()

# 新的身高、体重数据
test_height = np.array([140, 150, 145, 160])
test_weight = np.array([70, 60, 65, 75])
test_gender = np.array(['F', 'M', 'M', 'M'])

# 调用函数并传入相应的参数
plot_decision_boundary(height_data, weight_data, gender_data, degree=7, test_height=test_height, test_weight=test_weight, test_gender=test_gender)
