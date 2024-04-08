from extract_sift_features import extract_sift_features , normalize_descriptors_match
from sklearn.decomposition import PCA
from perceptron import perceptron
import numpy as np
def data_loader(d):
    # features的形状为 (n_samples, n_features)，这里假设n_samples为样本数，n_features为特征维度
    test_features = []
    train_features = []
    for i in range(100):
        Path = f'E:\\all_workspace\\ML\\Jupyter_notebook\\Pattern_recognition\\data\\dataset_homework5\\images\\test\\{i+1}.JPEG'
        points,test_img_future = extract_sift_features(Path)
        test_features.append(test_img_future)
    for i in range(400):
        Path = f'E:\\all_workspace\\ML\\Jupyter_notebook\\Pattern_recognition\\data\\dataset_homework5\\images\\train\\{i+1}.JPEG'
        points,trian_img_future = extract_sift_features(Path)
        train_features.append(trian_img_future)
    return train_features,test_features

def label_loader():
    # 读取txt文件，将每一行拆分成图片文件名和对应标签，并存储到字典中
    train_label = []
    test_label = []
    with open(r'E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\data\dataset_homework5\labels\test_label.txt', 'r') as file:
        for line in file:
            parts = line.split()  # 以空格分割每一行
            if len(parts) == 2:  # 确保每行有两个部分
                image_filename = parts[0]  # 第一个部分为图片文件名
                label = int(parts[1])  # 第二个部分为对应标签，转换为整数
                test_label.append(label)  # 将图片文件名和对应标签存储到字典中
    with open(r'E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\data\dataset_homework5\labels\train_label.txt', 'r') as file:
        for line in file:
            parts = line.split()  # 以空格分割每一行
            if len(parts) == 2:  # 确保每行有两个部分
                image_filename = parts[0]  # 第一个部分为图片文件名
                label = int(parts[1])  # 第二个部分为对应标签，转换为整数
                train_label.append(label)  # 将图片文件名和对应标签存储到字典中
    return train_label,test_label

if __name__ == '__main__':
    # 设置初始化w和b
    d = 100
    w0 = np.zeros(d)
    b0 = 0
    alpha = 0.2
    maxepoch = 100
    # 加载数据集和标签集
    train_set , test_set = data_loader(d)
    train_label,test_label = label_loader()

    # PCA：数据降维处理
    # test_features_normalized = normalize_descriptors_match(test_features)
    # train_features_normalized = normalize_descriptors_match(train_features)
    # 初始化PCA对象，设置降维后的目标维度为n_components
    n_components = d  # 设置目标维度为100
    pca = PCA(n_components=n_components)
    # 使用PCA对特征进行降维
    train_features_reduced = pca.fit_transform(train_set)
    test_features_reduced = pca.fit_transform(test_set)

    # 选取标签为0金鱼，标签为1青蛙的训练和测试集
    i =0
    test_data = [] # 测试集
    for value in test_label:
        if value == 0 or value ==1:
            test_data.append([test_set[i],value])
        i = i + 1
    train_data = [] #训练集
    i =0
    for label in train_label:
        if label ==0 or label ==1:
            train_data.append([train_set[i],label])
        i = i+1

    w,b = perceptron(train_data,w0,b0,alpha,maxepoch)
    success = 0
    # 在测试集上测试
    for x,y in test_data:
        if y*(np.dot(x,w))+b > 0:
            success = success +1
    accuracy = float(success/20)
    print('训练准确率：',accuracy)