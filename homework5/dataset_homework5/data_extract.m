%%
%图像数据预处理
clc;clear;
%%
%1.提取图像特征转换成特征向量
% 初始化特征列表
test_features = [];
train_features = [];

% 加载测试集图片
for i = 1:100
    Path = sprintf('E:\\all_workspace\\ML\\Jupyter_notebook\\Pattern_recognition\\homework5\\dataset_homework5\\images\\test\\%d.JPEG', i);
    vector = img2vector(Path);
    test_features = [test_features; vector];
end

% 加载训练集图片
for i = 1:400
    Path = sprintf('E:\\all_workspace\\ML\\Jupyter_notebook\\Pattern_recognition\\homework5\\dataset_homework5\\images\\train\\%d.JPEG', i);
    vector = img2vector(Path);
    train_features = [train_features; vector];
end
%%
% 使用PCA降维，将样本的特征向量从784维降到100维
[coeff_test, score,latent,tsquared, explained] = pca(double(test_features));
disp("主成分贡献：")
cumsum(latent)/sum(latent)%累积贡献值
cumulative_contribution = cumsum(latent) / sum(latent);
index_test = find(cumulative_contribution > 0.95, 1);%找到贡献达到95%的点
tran_matrix = coeff_test(:,1:80);
test_features_double = double(test_features);
test_features0 = bsxfun(@minus, test_features_double, mean(test_features_double, 1));
low_test_features = test_features0 * tran_matrix;
% 保存矩阵为.mat文件
save('test.mat', 'low_test_features');
% save('test.mat', 'test_features');

%%
[coeff_train, score,latent,tsquared, explained] = pca(double(train_features));
disp("主成分贡献：")
cumsum(latent)/sum(latent)%累积贡献值
cumulative_contribution = cumsum(latent) / sum(latent)
index_train = find(cumulative_contribution > 0.95, 1);%找到贡献达到95%的点
tran_matrix = coeff_train(:,1:80);
train_features_double = double(train_features);
train_features0 = bsxfun(@minus, train_features_double, mean(train_features_double, 1));
low_train_features = train_features0 * tran_matrix;
% 保存矩阵为.mat文件
save('train.mat', 'low_train_features');
% save('train.mat', 'train_features');

%%
%2.提取图像的对应标签
% 读取训练集标签文件
train_label_file = 'E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\labels\train_label.txt';
train_label_fid = fopen(train_label_file, 'r');
train_label = [];
% 逐行读取文件内容
tline = fgetl(train_label_fid);
while ischar(tline)
    parts = strsplit(tline);
    if numel(parts) == 2
        label = str2double(parts{2});
        train_label = [train_label, label];
    end
    tline = fgetl(train_label_fid);
end
fclose(train_label_fid);
% 保存矩阵为.mat文件
save('train_label.mat', 'train_label');
% 读取测试集标签文件
test_label_file = 'E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\labels\test_label.txt';
test_label_fid = fopen(test_label_file, 'r');
test_label = [];
% 逐行读取文件内容
tline = fgetl(test_label_fid);
while ischar(tline)
    parts = strsplit(tline);
    if numel(parts) == 2
        label = str2double(parts{2});
        test_label = [test_label, label];
    end
    tline = fgetl(test_label_fid);
end
fclose(test_label_fid);
% 保存矩阵为.mat文件
save('test_label.mat', 'test_label');

%%
%图像转向量函数
function vector = img2vector(Path)
    % 读取图片
    image = imread(Path);
    % 将图片转换为灰度图
    gray_image = im2gray(image);
    % 调整图片大小为28x28像素
    resized_image = imresize(gray_image, [28, 28]);

    % 将图片转换为向量
    vector = resized_image(:)';
end



