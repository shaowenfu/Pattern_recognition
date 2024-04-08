import cv2
import numpy as np


# 展示图像，封装成函数
def cv_show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，单位是毫秒，0代表任意键终止
    cv2.destroyAllWindows()


# 第一步：读取图像，咋换成灰度图
img = cv2.imread('1.JPEG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用cv2.xfeatures2d.SIFT_create() 实例化sift函数
sift = cv2.xfeatures2d.SIFT_create()

# 得到所有的关键点
keypoints = sift.detect(gray, None)  # 计算关键点
print(np.array(keypoints).shape)
keypoints, descriptor = sift.compute(gray, keypoints)  # 根据关键点计算周围区域的特征向量描述
print(np.array(keypoints).shape)
print(descriptor.shape)

# 或者一次性计算出关键点和特征向量，如下：
keypoints, descriptor = sift.detectAndCompute(gray, None)

# 打印特征点的信息，其中最关键的 kpoint.pt 就是包含了位置信息
for index, kpoint in enumerate(keypoints):
    print("关键点的位置是: pt[0]:{}\tpt[1]:{}".format(kpoint.pt[0], kpoint.pt[1]))

# 将关键点标记的图片上
img2 = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                         color=(255, 0, 0))

cv_show_image('SIFT', img2)

print('特征向量：',descriptor)