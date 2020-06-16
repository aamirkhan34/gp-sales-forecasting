import os
import selection
import variation
import prediction
import numpy as np
import pandas as pd


def min_max_train_test_scaling(X_train, X_test):
    # Min Max scaling approach
    alpha = 10.0

    X_train_scale = X_train.copy()
    X_test_scale = X_test.copy()

    for col in range(X_train.shape[1]):
        col_values = X_train[:, col]

        min_xi = np.min(col_values)
        max_xi = np.max(col_values)

        print(min_xi, max_xi)

        # Scaling on training data
        scaled_values_train = alpha*((col_values - min_xi) / (max_xi - min_xi))
        X_train_scale[:, col] = scaled_values_train

        # Scaling on test data using mean, std of train data
        scaled_values_test = alpha * \
            ((X_test[:, col] - min_xi) / (max_xi - min_xi))
        X_test_scale[:, col] = scaled_values_test

    return X_train_scale, X_test_scale


def standard_train_test_scaling(X_train, X_test):
    # Standard scaling approach
    alpha = 5.0

    X_train_scale = X_train.copy()
    X_test_scale = X_test.copy()

    for col in range(X_train.shape[1]):
        col_values = X_train[:, col]

        mean = np.mean(col_values)
        standard_deviation = np.std(col_values)

        # Scaling on training data
        scaled_values_train = alpha*((col_values - mean) / standard_deviation)
        X_train_scale[:, col] = scaled_values_train

        # Scaling on test data using mean, std of train data
        scaled_values_test = alpha * \
            ((X_test[:, col] - mean) / standard_deviation)
        X_test_scale[:, col] = scaled_values_test

    return X_train_scale, X_test_scale


def train_test_split(df, tdf, dataset):
    if "splitted_data" not in os.listdir():
        os.makedirs("splitted_data")

    if dataset not in os.listdir("splitted_data/"):
        os.makedirs("splitted_data/"+dataset)

    if "train.data" in os.listdir("splitted_data/"+dataset+"/"):
        train = pd.read_csv("splitted_data/"+dataset+"/train.data")
        test = pd.read_csv("splitted_data/"+dataset+"/test.data")
    else:
        df_copy = df.copy()

        train = df_copy.sample(frac=(1 - tdf/float(100)), random_state=0)
        test = df_copy.drop(train.index)

        train.to_csv("splitted_data/"+dataset+"/train.data", index=False)
        test.to_csv("splitted_data/"+dataset+"/test.data", index=False)

    print(train.values.shape, test.values.shape)

    return train.values[:, 0:-1], train.values[:, -1], test.values[:, 0:-1], test.values[:, -1]


def run_each_generation(program_list, generation_count, df, gap, tdp, MAX_PROGRAM_SIZE, dataset, NUMBER_OF_REGISTERS, vr_obj):
    X_train, y_train, X_test, y_test = train_test_split(df, tdp, dataset)

    # if dataset == "iris":
    #     X_train, X_test = min_max_train_test_scaling(X_train, X_test)
    # X_train, X_test = standard_train_test_scaling(X_train, X_test)

    # # print(pd.get_dummies(y_train))
    unique_classes = sorted(list(set(y_train)))
    register_class_map = {
        "R"+str(i): unique_classes[i] for i in range(len(unique_classes))}
    # print(register_class_map)
    train_accuracy = []
    fitness_scores = {}

    for gen in range(generation_count):
        print("Generation: ", gen+1)
        # Breeder model for selection-replacement
        fitness_scores = selection.get_fitness_scores(
            fitness_scores, program_list, vr_obj, X_train, y_train, register_class_map, gen, gap)

        program_list, fitness_scores = selection.rank_remove_worst_gap(
            gap, fitness_scores, program_list, register_class_map, dataset)
        train_accuracy.append(
            round((fitness_scores[list(fitness_scores.keys())[0]]/X_train.shape[0])*100, 2))
        selected_programs = selection.select_for_variation(gap, program_list)

        # Variations
        # Crossover
        offsprings = variation.crossover(
            selected_programs, gap, MAX_PROGRAM_SIZE)

        # Mutation
        offsprings = variation.mutation(offsprings)

        # Adding offsprings to program list
        program_list = program_list + offsprings
    print(train_accuracy)
    # Check saved model accuracy
    prediction_list = prediction.predict(
        X_test, "best_programs/"+dataset+"/best_program.json", NUMBER_OF_REGISTERS)
    if prediction_list:
        print("Accuracy: ", round(prediction.classifier_accuracy(
            prediction_list, y_test), 4))
