from scipy.io import loadmat, savemat

def data_loader():
    train_data = loadmat(r"E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\train.mat")['low_train_features']
    test_data = loadmat(r"E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\test.mat")['low_test_features']
    train_label = loadmat(r"E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\train_label.mat")['train_label']
    test_label = loadmat(r"E:\all_workspace\ML\Jupyter_notebook\Pattern_recognition\homework5\dataset_homework5\test_label.mat")['test_label']
    data_dim = train_data.shape[1]

    input_data_par = {}
    input_data_par['train_data'] = train_data
    input_data_par['text_data'] = test_data
    input_data_par['train_label'] = train_label
    input_data_par['test_label'] = test_label
    input_data_par['data_dim'] = data_dim

    return input_data_par