import csv
import statistics
import math


def read_file(file_name):
    with open(file_name) as csv_file:
        wine_data_reader = csv.reader(csv_file, delimiter=',')
        wine_data_records = []

        for record in wine_data_reader:
            record = list(map(float, record))
            wine_data_records.append(record)

        return wine_data_records


def probability(x, data_by_class, total_number_of_records):
    most_probable_result = [-1, -1]

    for i in range(len(data_by_class)):
        probability_of_x_in_class = len(data_by_class[i]) / total_number_of_records
        class_of_data = data_by_class[i][0][0]

        for i1 in range(1, len(data_by_class[i][0])):
            every_record_in_class_attribute = []

            for i2 in range(len(data_by_class[i])):
                every_record_in_class_attribute.append(data_by_class[i][i2][i1])

            mean = statistics.mean(every_record_in_class_attribute)
            stdev = statistics.stdev(every_record_in_class_attribute)
            probability_for_attribute = (1 / (stdev * math.sqrt(2 * math.pi))) * math.exp((-(x[i1] - mean) ** 2) / (2 * stdev ** 2))
            probability_of_x_in_class = probability_of_x_in_class * probability_for_attribute

        if probability_of_x_in_class > most_probable_result[1]:
            most_probable_result = [class_of_data, probability_of_x_in_class]

    return most_probable_result[0]


def n_cross_validation(records, attributes_id):
    predicted_correct = 0
    predicted_wrong = 0

    for i in range(len(records)):
        train_set_all_attributes = records[:i] + records[i+1:]
        test_set_all_attributes = records[i]
        train_set = []

        for train_set_record in train_set_all_attributes:
            train_set_selected_attributes_record = [train_set_record[0]]

            for i1 in range(len(attributes_id)):
                train_set_selected_attributes_record.append(train_set_record[attributes_id[i1]])

            train_set.append(train_set_selected_attributes_record)

        test_set = [test_set_all_attributes[0]]

        for i1 in range(len(attributes_id)):
            test_set.append(test_set_all_attributes[attributes_id[i1]])

        train_set_grouped_by_class = []
        list_of_class_in_train_set = []

        for record in train_set:
            is_record_of_new_class = True
            class_of_record = record[0]

            for i1 in range(len(list_of_class_in_train_set)):
                if class_of_record == list_of_class_in_train_set[i1]:
                    is_record_of_new_class = False
                    class_in_list_id = i1
                    train_set_grouped_by_class[class_in_list_id].append(record)
                    break

            if is_record_of_new_class:
                unique_class_list = [record]
                train_set_grouped_by_class.append(unique_class_list)
                list_of_class_in_train_set.append(class_of_record)

        predicted_class_of_test_set = probability(test_set, train_set_grouped_by_class, len(train_set_all_attributes))

        if predicted_class_of_test_set == test_set[0]:
            predicted_correct += 1
        else:
            predicted_wrong += 1

    accuracy_of_predictions = (predicted_correct / (predicted_correct + predicted_wrong)) * 100
    return accuracy_of_predictions


results_for_attributes = []
file_name = str(input("Type file name(e.g. wine.data): "))
wine_data = read_file(file_name)
for a in range(1, 14):
    results_for_attributes.append([a, round(n_cross_validation(wine_data, [a]), 2)])

results_for_attributes.sort(key=lambda x: x[1], reverse=True)
print("List of attributes with the accuracy of their classification:")
print(*results_for_attributes, sep='\n')
