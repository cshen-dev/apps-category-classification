from time import ctime

from algorithm.evaluate import classify
from algorithm.naive_bayes import *

from algorithm.io_utils import *


def main():
    print("==> Main app start at:", ctime())

    print("==> File reading phase start at:", ctime())
    data_list = read_training_data('../input/training_data.csv')
    labels_list = read_training_labels('../input/training_labels.csv')
    test_list = read_test_data('../input/test_data.csv')
    print("<== File reading phase end at:", ctime())

    print("==> training phase start at:", ctime())
    classes = stat_classes(labels_list)
    ratio = cal_class_ratio(labels_list, classes)
    probs = cal_prob_classes(data_list, labels_list, classes)
    print("<== training phase end at :", ctime())

    print("==> test phase start at :", ctime())
    results = classify(test_list, probs, classes, ratio)
    print("<== test phase end at :", ctime())
    output_test_result(results, '../output/predicted_labels.csv')
    print("<== Main app end at:", ctime())


if __name__ == "__main__":
    main()
