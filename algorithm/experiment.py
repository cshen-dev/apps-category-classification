from evaluate import *
from naive_bayes import *
from io_utils import *


def run_one_loop(tr_data, tr_labels, splits, loop_num):
    test_data = tr_data[splits[loop_num][0]:splits[loop_num][1]]
    test_labels = tr_labels[splits[loop_num][0]:splits[loop_num][1]]
    train_data = []
    train_labels = []

    for i in range(len(splits)):
        if i == loop_num:
            continue
        train_data.extend(tr_data[splits[i][0]: splits[i][1]])
        train_labels.extend(tr_labels[splits[i][0]: splits[i][1]])

    # print("test check:", check_entry_order(test_data, test_labels))
    # print("train check:", check_entry_order(train_data, train_labels))
    # print("train_data", len(train_data))
    # print("train_labels", len(train_labels))

    print("== current loop:", loop_num)
    print("==> stat_classes phase start at:", ctime())
    classes = stat_classes(train_labels)
    print("<== stat_classes phase end at :", ctime())
    print(classes)

    print("==> cal_class_ratio phase start at :", ctime())
    ratio = cal_class_ratio(train_labels, classes)
    print("<== cal_class_ratio phase end at :", ctime())
    print(ratio)

    print("==> cal_prob_classes phase start at :", ctime())
    probs = cal_prob_classes(train_data, train_labels, classes)
    print("<== cal_prob_classes phase end at :", ctime())
    print(probs)

    print("==> cal_confusion_matrix phase start at :", ctime())
    confusion_matrix = stat_confusion_matrix(test_data, test_labels, probs, classes, ratio)
    print("<== cal_confusion_matrix end at :", ctime())
    return confusion_matrix, classes


def main():
    print("==> Main app start at:", ctime())

    print("==> File reading phase start at:", ctime())
    data_list = read_training_data('../input/training_data.csv')
    labels_list = read_training_labels('../input/training_labels.csv')
    print("<== File reading phase end at:", ctime())

    # check = check_entry_order(data_list, labels_list)
    # print(check)

    # for cross validation 10-fold
    folds = 10
    row_num = 20104
    interval = round(row_num / folds)
    splits = []
    classes = None

    for i in range(folds):
        if i == folds - 1:
            splits.append((i * interval, row_num))
            continue
        splits.append((i*interval, (i+1)*interval))

    confusion_matrix = np.zeros((30, 30), dtype=int)
    for i in range(0, folds):
        confusion_matrix_tmp, classes_tmp = run_one_loop(data_list, labels_list, splits, i)
        confusion_matrix += confusion_matrix_tmp
        classes = classes_tmp

    print("confusion_matrix:", confusion_matrix/folds)
    sensitive_measure = cal_cost_sensitive_measure(confusion_matrix, classes)
    print("sensitive_measure:", sensitive_measure)
    print("<== Main app end at:", ctime())


if __name__ == "__main__":
    main()







