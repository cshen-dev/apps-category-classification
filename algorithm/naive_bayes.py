import numpy as np

threshold = 0


def stat_classes(labels):
    unique_set = set()
    for entry in labels:
        unique_set.add(entry[1])
    unique_list = list(unique_set)
    unique_list.sort()
    return unique_list


def cal_prob_classes(data, labels, classes):
    classes_prob_list = np.ones((len(classes), len(data[0])-1))
    classes_count_list = np.array([2]*len(classes)*(len(data[0])-1)).reshape((len(classes), len(data[0])-1))

    for entry in zip(data, labels):
        class_name = entry[1][1]
        class_num = classes.index(class_name)

        for i in range(1, len(entry[0])):
            if float(entry[0][i]) > threshold:
                classes_prob_list[class_num][i-1] += 1
            classes_count_list[class_num][i-1] += 1

    classes_prob_list = classes_prob_list / classes_count_list
    classes_prob_inverse_list = np.ones_like(classes_prob_list) - classes_prob_list
    return np.log(classes_prob_list), np.log(classes_prob_inverse_list)


def cal_class_ratio(labels, classes):
    count = np.zeros((len(classes)))

    for label in labels:
        cls_index = classes.index(label[1])
        count[cls_index] += 1

    return np.log(count/len(labels))


def estimate_feature(test, prob_classes, classes, classes_ratio):
    final_probs = [0] * len(classes)

    for i in range(len(classes)):
        prob_upper = 0.0
        for j in range(1, len(test)):
            if float(test[j]) > threshold:
                prob_upper += prob_classes[0][i][j-1]
            else:
                prob_upper += prob_classes[1][i][j-1]
        final_probs[i] = prob_upper + classes_ratio[i]

    index = max([(v, i) for i, v in enumerate(final_probs)])[1]
    return classes[index]







