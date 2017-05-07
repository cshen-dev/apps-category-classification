from operator import itemgetter
import csv


def read_training_data(file_name):
    output = []
    with open(file_name, 'r') as data:
        data_reader = csv.reader(data)
        for i, row in enumerate(data_reader):
            data_entry = []
            for j in row:
                data_entry.append(j)
            output.append(data_entry)
    output.sort(key=itemgetter(0))
    return output


def read_training_labels(file_name):
    output = []
    with open(file_name, 'r') as labels:
        labels_reader = csv.reader(labels)
        for i, row in enumerate(labels_reader):
            labels_entry = (row[0], row[1])
            output.append(labels_entry)
    output.sort(key=itemgetter(0))
    return output


def check_entry_order(left_list, right_list):
    count = 0
    for row in zip(left_list, right_list):
        l_name = row[0][0]
        r_name = row[1][0]
        if l_name == r_name:
            count += 1
    return count


def read_test_data(file_name):
    output = []
    with open(file_name, 'r') as data:
        data_reader = csv.reader(data)
        for i, row in enumerate(data_reader):
            data_entry = []
            for j in row:
                data_entry.append(j)
            output.append(data_entry)
    return output


def output_test_result(data, file_name):
    with open(file_name, 'w') as output:
        data_writer = csv.writer(output)
        data_writer.writerows(data)
    return

