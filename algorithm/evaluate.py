from time import ctime
import numpy as np
import naive_bayes as nb


def classify(data, prob_classes, classes, classes_ratio):
    results = []

    for i in range(0, len(data)):
        result = nb.estimate_feature(data[i], prob_classes, classes, classes_ratio)
        results.append((data[i][0], result))
    return results


def cal_correctness(data, label, prob_classes, classes, classes_ratio):
    correct = 0
    total = 0
    for i in range(0, len(data)):
        result = nb.estimate_feature(data[i], prob_classes, classes, classes_ratio)
        if result == label[i][1]:
            correct += 1
        total += 1
        if i == round(len(data)/5):
            print("20% finished at", ctime())
        elif i == round(len(data)/3):
            print("33% finished at", ctime())
        elif i == round(len(data) / 2):
            print("50% finished at", ctime())
        elif i == round(len(data) / 1.5):
            print("66% finished at", ctime())
        elif i == round(len(data) / 1.25):
            print("80% finished at", ctime())
        elif i == round(len(data) / 1):
            print("100% finished", ctime())
    return correct / total


def stat_confusion_matrix(data, label, prob_classes, classes, classes_ratio):
    class_num = len(classes)
    confusion_matrix = np.zeros((class_num, class_num), dtype=int)

    for i in range(0, len(data)):
        result = nb.estimate_feature(data[i], prob_classes, classes, classes_ratio)

        confusion_matrix[classes.index(label[i][1])][classes.index(result)] += 1

        if i == round(len(data)/5):
            print("20% finished at", ctime())
        elif i == round(len(data)/3):
            print("33% finished at", ctime())
        elif i == round(len(data) / 2):
            print("50% finished at", ctime())
        elif i == round(len(data) / 1.5):
            print("66% finished at", ctime())
        elif i == round(len(data) / 1.25):
            print("80% finished at", ctime())
        elif i == round(len(data) / 1):
            print("100% finished", ctime())

    # print("sum of confusion matrix", confusion_matrix.sum())
    return confusion_matrix


def cal_cost_sensitive_measure(confusion_matrix, classes):
    result = []

    for i, class_item in enumerate(classes):
        result_item = dict()

        result_item["name"] = class_item

        tp = confusion_matrix[i][i]
        result_item["TP"] = tp

        fp = 0
        for r_i, row in enumerate(confusion_matrix):
            if r_i == i:
                continue
            fp += row[i]
        result_item["FP"] = fp

        fn = 0
        for c_i, col in enumerate(confusion_matrix[i]):
            if c_i == i:
                continue
            fn += col
        result_item["FN"] = fn

        tn = 0
        for r_i, row in enumerate(confusion_matrix):
            if r_i == i:
                continue
            for c_i, col in enumerate(confusion_matrix[r_i]):
                if c_i == i:
                    continue
                tn += col
        result_item["TN"] = tn

        accuracy = (tp+tn)/(tp+fn+fp+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f_measure = 2 * tp / (2 * tp + fn + fp)

        result_item["accuracy"] = accuracy
        result_item["precision"] = precision
        result_item["recall"] = recall
        result_item["f_measure"] = f_measure
        result.append(result_item)

    return result


