from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    # Initial shuffle
    indices = []
    for i in range(labels.size):
        indices.append(i)
    indices = permutation(indices)
    data = data[indices]
    labels = labels[indices]

    # Limiting the data to max_count
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    # Splitting the data
    train_number_of_instances = int(train_ratio * data.shape[0])
    train_data = data[0:train_number_of_instances, :]
    train_labels = labels[0:train_number_of_instances]
    test_data = data[train_number_of_instances:data.size, :]
    test_labels = labels[train_number_of_instances:data.size]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(labels.size):
        if prediction[i] == 1 and labels[i] == 1:
            tp += 1
        if prediction[i] == 1 and labels[i] == 0:
            fp += 1
        if prediction[i] == 0 and labels[i] == 0:
            tn += 1
        if prediction[i] == 0 and labels[i] == 1:
            fn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    for i in range(len(folds_array)):
        train = concatenate([x for j, x in enumerate(folds_array) if j != i])
        train_labels = concatenate([x for j, x in enumerate(labels_array) if j != i])
        test = folds_array[i]
        test_labels = labels_array[i]
        clf.fit(train, train_labels)
        prediction_labels = clf.predict(test)
        current_tpr, current_fpr, current_accuracy = get_stats(prediction_labels, test_labels)
        tpr.append(current_tpr)
        fpr.append(current_fpr)
        accuracy.append(current_accuracy)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params

    # Splitting the data and labels
    data_array = array_split(data_array, folds_count)
    labels_array = array_split(labels_array, folds_count)
    # Storing the data in a list
    tpr = []
    fpr = []
    accuracy = []

    for i in range(len(kernels_list)):
        clf = None
        kernel = kernels_list[i]
        c = kernel_params[i]['C'] if kernel_params[i].get('C') else 1
        degree = kernel_params[i]['degree'] if kernel_params[i].get('degree') else 3
        gamma = kernel_params[i]['gamma'] if kernel_params[i].get('gamma') else 'auto'

        clf = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)

        current_tpr, current_fpr, current_accuracy = get_k_fold_stats(data_array, labels_array, clf)
        tpr.append(current_tpr)
        fpr.append(current_fpr)
        accuracy.append(current_accuracy)

    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy

    return svm_df


def get_most_accurate_kernel(accuracy):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    return accuracy.idxmax()


def get_kernel_with_highest_score(score):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    return score.idxmax()


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    highest_score_index = get_kernel_with_highest_score(df.score)
    plt.scatter(x, y)
    plt.plot(x[highest_score_index], alpha_slope * x[highest_score_index])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


def evaluate_c_param(data_array, labels_array, folds_count, kernel, kernel_params):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernel: the kernel to use
    :param kernel_params: the kernel parameters to use
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """
    kernel_params_with_c = []
    kernels = []
    for i in range(-4, 1):
        for j in range(1, 3):
            params_dict = {'C': (10 ** i) * (j / 3)}
            params_dict.update(kernel_params)
            kernel_params_with_c.append(params_dict)
            kernels.append(kernel)

    res = compare_svms(data_array, labels_array, folds_count,
                       kernels_list=tuple(kernels), kernel_params=tuple(kernel_params_with_c))

    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, kernel_type, kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test

    :param kernel_type: the chosen kernel
    :param kernel_params: the chosen kernel parameters
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    c = kernel_params['C'] if kernel_params.get('C') else 1
    degree = kernel_params['degree'] if kernel_params.get('degree') else 3
    gamma = kernel_params['gamma'] if kernel_params.get('gamma') else 'auto'

    clf = SVC(class_weight='balanced', C=c, degree=degree, gamma=gamma)  # TODO: set the right kernel
    clf.fit(train_data, train_labels)
    test_prediction = clf.predict(test_data)

    tpr, fpr, accuracy = get_stats(test_prediction, test_labels)

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
