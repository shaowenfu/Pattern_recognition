import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# 已知数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])
gender = np.array(['M', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M'])

# 分离男女数据
male_height = height_data[gender == 'M']
male_weight = weight_data[gender == 'M']
female_height = height_data[gender == 'F']
female_weight = weight_data[gender == 'F']

# 绘制散点图
plt.scatter(male_height, male_weight, color='blue', label='Male')
plt.scatter(female_height, female_weight, color='pink', label='Female')

# 设置图形属性
plt.title('Distribution of Gender based on Height and Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
