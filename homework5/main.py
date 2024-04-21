from data_loader import data_loader
from models import Perceptron, LSC, Fisher, Multi_cla

if __name__ == '__main__':
    # 加载数据集和标签集
    input_data_par = data_loader()
    train_set = input_data_par['train_data']
    test_set = input_data_par['text_data']
    train_label = input_data_par['train_label'].flatten()
    test_label = input_data_par['test_label'].flatten()
    data_dim = input_data_par['data_dim']
    print("数据维度：", data_dim)
    Model = 'Multi_cla'  # 指定使用的分类准则 perceptron、LSC、Fisher、Multi_cla

    # 选择模型
    if Model == 'perceptron':
        model = Perceptron(test_label, test_set, train_label, train_set, data_dim)

    elif Model == 'LSC':
        model = LSC(test_label, test_set, train_label, train_set, data_dim)

    elif Model == 'Fisher':
        model = Fisher(test_label, test_set, train_label, train_set, data_dim)

    elif Model == 'Multi_cla':
        model = Multi_cla(test_label, test_set, train_label, train_set, data_dim)

    # 训练和测试
    model.fuc()
