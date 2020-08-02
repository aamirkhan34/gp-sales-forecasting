import os

# Third party imports
import numpy as np
import pandas as pd

# Application imports
import scaling
import selection
import variation
import prediction
from detection_tracking import DetectionTracking


def validate_get_register_count(NUMBER_OF_REGISTERS, df):
    number_of_classes = len(set(df.values[:, -1].tolist()))
    print("Number of labels: ", number_of_classes)
    if number_of_classes > NUMBER_OF_REGISTERS:
        NUMBER_OF_REGISTERS = number_of_classes + 1

    if NUMBER_OF_REGISTERS > 20:
        NUMBER_OF_REGISTERS = 20
    print("Number of registers: ", NUMBER_OF_REGISTERS)
    return NUMBER_OF_REGISTERS


def load_dataset(name_of_dataset):
    df = pd.read_csv(name_of_dataset+"/"+name_of_dataset+".data", header=None)
    # Drop columns with all null values
    df = df.dropna(axis=1, how='all')
    return df


def preprocess_data(df):
    return df


def get_scaled_data(X_train, X_test):
    X_train, X_test = scaling.min_max_train_test_scaling(X_train, X_test)
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
        df_copy = df.copy()

        train = df_copy.sample(frac=(1 - tdp/float(100)), random_state=0)
        test = df_copy.drop(train.index)

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

    for gen in range(generation_count):
        print("Generation: ", gen+1)

        # Breeder model for selection-replacement
        fitness_scores = selection.get_fitness_scores(
            fitness_scores, program_list, vr_obj, X_train, y_train, register_class_map, gen, gap, False, dt_obj)

        program_list, fitness_scores = selection.rank_remove_worst_gap(
            gap, fitness_scores, program_list, register_class_map, dataset)

        # Update detection tracker for best individual
        fit_score_best = selection.get_fitness_score_of_program(
            program_list[0], vr_obj, X_train, y_train, register_class_map, gen, dt_obj, True)

        train_accuracy.append(
            round((fitness_scores[list(fitness_scores.keys())[0]]/X_train.shape[0])*100, 2))

        selected_programs = selection.select_for_variation(gap, program_list)

        # Variations

        # Crossover
        offsprings = variation.crossover(
            selected_programs, gap, MAX_PROGRAM_SIZE)

        # Mutation
        offsprings = variation.mutation(
            offsprings, source_x, target_r, source_r, ops, vr_obj, X_train, y_train, register_class_map)

        # Adding offsprings to program list
        program_list = program_list + offsprings

    print(train_accuracy)

    return train_accuracy, program_list, dt_obj


def save_and_test_champ_classifier(program_list, X_train, y_train, X_test, y_test, register_class_map,
                                   NUMBER_OF_REGISTERS, dataset, st):
    # Check saved model accuracy - test data
    prediction_list = prediction.predict(
        X_test, "fittest_programs/"+dataset+"/fittest_program.json", NUMBER_OF_REGISTERS)

    if prediction_list:
        print("Accuracy for Champ classifier(Accuracy): ", round(prediction.classifier_accuracy(
            prediction_list, y_test), 4))
