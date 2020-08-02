import os

# Third party imports
import numpy as np
import pandas as pd

# Application imports
import scaling
import sampling
import selection
import variation
import prediction
from detection_tracking import DetectionTracking


def validate_get_register_count(NUMBER_OF_REGISTERS, df):
    number_of_classes = len(set(df.values[:, -1].tolist()))
    print("Number of classes: ", number_of_classes)
    if number_of_classes > NUMBER_OF_REGISTERS:
        NUMBER_OF_REGISTERS = number_of_classes + 1

    if NUMBER_OF_REGISTERS > 20:
        NUMBER_OF_REGISTERS = 20

    return NUMBER_OF_REGISTERS


def load_dataset(name_of_dataset):
    df = pd.read_csv(name_of_dataset+"/ann-train.data", sep=" ", header=None)
    # Drop columns with all null values
    df = df.dropna(axis=1, how='all')
    return df


def preprocess_data(df):
    return df


def get_scaled_data(X_train, X_test):
    # X_train, X_test = scaling.min_max_train_test_scaling(X_train, X_test)
    X_train, X_test = scaling.standard_train_test_scaling(X_train, X_test)
    # X_train, X_test = scaling.sklearn_standard_scaling(X_train, X_test)
    return X_train, X_test


def train_test_split(df, tdp, dataset):
    if "splitted_data" not in os.listdir():
        os.makedirs("splitted_data")

    if dataset not in os.listdir("splitted_data/"):
        os.makedirs("splitted_data/"+dataset)

    if "train.data" in os.listdir("splitted_data/"+dataset+"/"):
        train = pd.read_csv("splitted_data/"+dataset+"/train.data")
        test = pd.read_csv("splitted_data/"+dataset+"/test.data")
    else:
        train = pd.read_csv(dataset+"/ann-train.data", sep=" ", header=None)
        # Drop columns with all null values
        train = train.dropna(axis=1, how='all')

        test = pd.read_csv(dataset+"/ann-test.data", sep=" ", header=None)
        # Drop columns with all null values
        test = test.dropna(axis=1, how='all')

        train.to_csv("splitted_data/"+dataset+"/train.data", index=False)
        test.to_csv("splitted_data/"+dataset+"/test.data", index=False)

    print(train.values.shape, test.values.shape)

    return train.values[:, 0:-1], train.values[:, -1], test.values[:, 0:-1], test.values[:, -1]


def train_gp_classifier(generation_count, program_list, train_accuracy, fitness_scores,
                        vr_obj, X_train, y_train, register_class_map, gap, dataset,
                        MAX_PROGRAM_SIZE, t, rp, st, NUMBER_OF_REGISTERS):
    # Detection tracking
    dt_obj = DetectionTracking(list(register_class_map.values()))

    # Attributes of dataset
    source_x = list(range(0, X_train.shape[1] - 1))

    # Registers
    target_r = list(range(0, NUMBER_OF_REGISTERS))
    source_r = list(range(0, NUMBER_OF_REGISTERS))

    # Operators
    ops = [0, 1, 2, 3]

    # Sampling function mapper
    sampling_mapper = {"uniformly": sampling.uniform_sampling,
                       "equally": sampling.balanced_uniform_sampling}
    sample_flag = False

    for gen in range(generation_count):
        print("Generation: ", gen+1)

        if (gen % rp) == 0:
            print("Sampling..")
            X_train_t, y_train_t = sampling_mapper[st](X_train, y_train, t)
            sample_flag = True

        # Breeder model for selection-replacement
        fitness_scores = selection.get_fitness_scores(
            fitness_scores, program_list, vr_obj, X_train_t, y_train_t, register_class_map, gen, gap, sample_flag, dt_obj)

        sample_flag = False

        program_list, fitness_scores = selection.rank_remove_worst_gap(
            gap, fitness_scores, program_list, register_class_map, dataset)

        # Update detection tracker for best individual
        fit_score_best = selection.get_fitness_score_of_program(
            program_list[0], vr_obj, X_train_t, y_train_t, register_class_map, gen, dt_obj, True)

        train_accuracy.append(
            round((fitness_scores[list(fitness_scores.keys())[0]]/X_train_t.shape[0])*100, 2))

        selected_programs = selection.select_for_variation(gap, program_list)

        # Variations

        # Crossover
        offsprings = variation.crossover(
            selected_programs, gap, MAX_PROGRAM_SIZE)

        # Mutation
        offsprings = variation.mutation(
            offsprings, source_x, target_r, source_r, ops, vr_obj, X_train_t, y_train_t, register_class_map)

        # Adding offsprings to program list
        program_list = program_list + offsprings

    print(train_accuracy)

    return train_accuracy, program_list, dt_obj


def save_and_test_champ_classifier(program_list, X_train, y_train, X_test, y_test, register_class_map,
                                   NUMBER_OF_REGISTERS, dataset, st):
    # Check and save best classifier - training data
    prediction.predict_and_save_best_classifier(
        program_list, X_train, y_train, register_class_map, NUMBER_OF_REGISTERS, dataset, st)

    # Check saved model accuracy - test data
    prediction_list1 = prediction.predict(
        X_test, "best_programs/"+dataset+"/"+st+"/Accuracy/best_program.json", NUMBER_OF_REGISTERS)

    # Check saved model detectionRate - test data
    prediction_list2 = prediction.predict(
        X_test, "best_programs/"+dataset+"/"+st+"/DetectionRate/best_program.json", NUMBER_OF_REGISTERS)

    if prediction_list1:
        print("Accuracy of Champ classifier(Accuracy): ", round(prediction.classifier_accuracy(
            prediction_list1, y_test), 4))

        prediction.save_confusion_matrix(
            y_test, prediction_list1, dataset, st, "accuracy")

    if prediction_list2:
        print("Accuracy of Champ classifier(DetectionRate): ", round(prediction.classifier_accuracy(
            prediction_list2, y_test), 4))

        prediction.save_confusion_matrix(
            y_test, prediction_list2, dataset, st, "detection_rate")
