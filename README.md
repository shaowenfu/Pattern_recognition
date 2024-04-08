模式识别课程的课后作业的个人解答
# 作业一
针对课件中统计模式识别实例，找出一个你认为最好的线性分界面（线），并计算出相应的识别准确率（假设未知样本中只有第一个实际为女性）。
在问题3中特征空间是二维的（身高、体重）。如果分别考虑身高和体重两个属性，请找出相应的一维特征空间中的最佳分界面（点），并计算其识别准确率。
19名男女同学进行体检，测量了身高和体重，但事后发现其中有4人忘记填写性别，试问这4人是男是女？
<img width="650" alt="image" src="https://github.com/shaowenfu/Pattern_recognition/assets/129816349/42a22d14-6e5d-42be-86d7-3c5ebcf6e4f6">
待识别的模式类：性别（男或女）
测量的特征：身高和体重
训练样本：15名已知性别的样本特征
目标：希望借助于训练样本的特征建立判别函数（即数学模型），从而对未知样本作出识别（分类）
从图中训练样本的分布情况，找出男、女两类特征各自的聚类特点，从而求取一个分界面（直线或曲线），即判别函数
根据待识别样本在判别函数的哪一侧来识别性别
<img width="448" alt="image" src="https://github.com/shaowenfu/Pattern_recognition/assets/129816349/4643d7bc-8c07-4e71-b2f6-f0061bf1b5cb">
# 作业三
推导以下四种贝叶斯决策规则。
最小错误率贝叶斯决策规则
最小风险贝叶斯决策规则
最小最大贝叶斯决策规则（选做）
限定错误率贝叶斯决策规则（选做）
参照课件中癌细胞分类的例子，自己设计一个新例子，并根据上述四种贝叶斯决策规则分别构造相应的分类器，给出分类结果。
提示：根据规则作必要的假设。
# 作业五
推导感知器准则、最小平方误差准则。
从群文件下载动物识别数据集，完成以下实验（使用语言不限）：
实现感知器准则、最小平方误差准则和Fisher准则，并针对类别金鱼和青蛙的识别进行评估实验；
基于本讲和上一讲介绍的准则实现一种多类分类器，针对动物识别数据集中的所有类别进行评估实验。
数据集包含：10个类别各50张64*64的图片和包含对应标签的txt文档。测试集:训练集=1:4。