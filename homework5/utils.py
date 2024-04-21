import numpy as np
import matplotlib.pyplot as plt
class Weight:
    def __init__(self, left_index, right_index, left=None, right=None, data=None, b=None):
        self.left = left
        self.right = right
        self.left_index  = left_index
        self.right_index = right_index
        self.data = data
        self.b = b


# 求伪逆矩阵
def inverse_matrix(data):
    T_data = transpose(data)
    tmp_data = np.dot(T_data,data)
    # print("tmp_data:",tmp_data)
    inverse_data = np.linalg.inv(tmp_data)
    # print("inverse_data:",inverse_data)
    result_data = np.dot(inverse_data, T_data)
    # print("result_data", result_data)
    return result_data

def transpose(matrix):
    row, col = len(matrix), len(matrix[0])
    new_matrix = []
    for j in range(col):
        row_ = []
        for i in range(row):
            row_.append(matrix[i][j])

        new_matrix.append(row_)
    # print(new_matrix)
    return new_matrix

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def VGG_extract_features_from_image(image_path):
    # 加载预训练的 VGG 模型
    vgg_model = models.vgg16(pretrained=True)

    # 冻结模型的参数
    for param in vgg_model.parameters():
        param.requires_grad = False

    # 加载图像并进行预处理
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    # 提取图像的特征并编码为特征向量
    with torch.no_grad():
        vgg_model.eval()
        features = vgg_model(image_tensor)
    features = features.squeeze().numpy()

    return features

# # 示例用法
# image_path = '1.JPEG'
# features = VGG_extract_features_from_image(image_path)
# print("Features:", len(features))

import torch.nn as nn
import torch.nn.functional as F


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out