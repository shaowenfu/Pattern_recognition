import cv2
import numpy as np

# 特征向量对齐---填充和截断
def normalize_descriptor_lenth(descriptor_flattened, desired_length):
    current_length = len(descriptor_flattened)

    if current_length < desired_length:
        # 如果特征向量长度小于期望长度，则在末尾填充0，直到达到期望长度
        padding = np.zeros(desired_length - current_length)
        normalized_descriptor = np.concatenate((descriptor_flattened, padding))
    elif current_length > desired_length:
        # 如果特征向量长度大于期望长度，则截断到期望长度
        normalized_descriptor = descriptor_flattened[:desired_length]
    else:
        # 如果特征向量长度已经等于期望长度，则直接返回
        normalized_descriptor = descriptor_flattened

    return normalized_descriptor



# 特征向量对齐---特征值匹配
def normalize_descriptors_match(descriptors_list):
    # 选择第一个特征向量作为基准
    base_descriptor = descriptors_list[0]

    # 使用 Brute-Force 匹配器进行特征点匹配
    bf = cv2.BFMatcher()
    matches = []
    for descriptor in descriptors_list[1:]:
        matches.append(bf.match(base_descriptor, descriptor))

    # 根据匹配结果对特征向量进行对齐和标准化
    normalized_descriptors = []
    for match, descriptor in zip(matches, descriptors_list[1:]):
        # 提取匹配对的特征点
        src_pts = np.float32([base_descriptor[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([descriptor[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

        # 计算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 使用变换矩阵对特征向量进行对齐
        descriptor_aligned = cv2.perspectiveTransform(descriptor, M)

        # 将对齐后的特征向量添加到列表中
        normalized_descriptors.append(descriptor_aligned.flatten())

    # 返回对齐后的特征向量列表
    return normalized_descriptors

def extract_sift_features(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT 创建对象
    sift = cv2.SIFT_create()

    # 计算关键点和特征向量
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 将特征向量展开成一维数组
    descriptor_flattened = descriptors.flatten() if descriptors is not None else np.array([])

    # print("Number of keypoints:", len(keypoints))
    # print("Flattened descriptor shape:", descriptor_flattened.shape)
    descriptor_flattened = normalize_descriptor_lenth(descriptor_flattened,10000)
    return keypoints, descriptor_flattened

# 示例用法
# image_path = r'E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\data\dataset_homework5\images\test\1.JPEG'
# keypoints, descriptor_flattened = extract_sift_features(image_path)



