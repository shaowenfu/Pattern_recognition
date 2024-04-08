import numpy as np
def fuc(data,string):
    gender_data = np.array(['M', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M'])
    # 将性别转换为数字标签，M: 0, F: 1
    gender_labels = np.where(gender_data == 'M', 0, 1)

    feature = data

    # 对数据进行排序
    sorted_indices = np.argsort(feature)
    sorted_feature = feature[sorted_indices]
    sorted_labels = gender_labels[sorted_indices]

    # 尝试不同的阈值
    accuracies = []
    for i in range(1, len(sorted_feature)):
        threshold = (sorted_feature[i - 1] + sorted_feature[i]) / 2
        predicted_labels = np.where(sorted_feature <= threshold, 0, 1)
        accuracy = np.mean(predicted_labels == sorted_labels)
        accuracies.append(accuracy)

    # 找到最佳阈值
    best_threshold = (sorted_feature[np.argmax(accuracies) - 1] + sorted_feature[np.argmax(accuracies)]) / 2
    best_accuracy = np.max(accuracies)

    print(f"{string}分界线: {best_threshold}")
    print(f"准确率: {best_accuracy}")


# 数据
height_data = np.array([170, 130, 180, 190, 160, 150, 190, 210, 100, 170, 140, 150, 120, 150, 130])
weight_data = np.array([68, 66, 71, 73, 70, 66, 68, 76, 58, 75, 62, 64, 66, 66, 65])

fuc(height_data,"身高")
fuc(weight_data,"体重")
