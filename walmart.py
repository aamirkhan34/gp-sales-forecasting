import os

# Third party imports
import numpy as np
import pandas as pd
from sklearn import model_selection

# Application imports
import scaling
import sampling
import selection
import variation
import prediction
# from detection_tracking import DetectionTracking
from walmart_preprocessing import WalmartPreprocessingRegression, WalmartPreprocessingForecasting


def validate_get_register_count(NUMBER_OF_REGISTERS):
    if NUMBER_OF_REGISTERS < 1:
        NUMBER_OF_REGISTERS = 4

    return NUMBER_OF_REGISTERS


def load_dataset(path):
    df = pd.read_csv(path)
    # Drop columns with all null values
    df = df.dropna(axis=1, how='all')

    return df


def save_dataframe_to_csv(df, dataset, prediction_type, filename):
    if "splitted_data" not in os.listdir():
        os.makedirs("splitted_data")

    if dataset not in os.listdir("splitted_data/"):
        os.makedirs("splitted_data/"+dataset)

    if prediction_type not in os.listdir("splitted_data/"+dataset):
        os.makedirs("splitted_data/"+dataset+"/"+prediction_type)

    df.to_csv("splitted_data/"+dataset+"/" +
              prediction_type+"/"+filename, index=False)


def move_ycolumn_at_end(df, col):
    prediction_col = df[col]

    # Delete column
    df.drop(labels=[col], axis=1, inplace=True)

    # Insert col at end
    df.insert(len(df.columns.values), col, prediction_col)

    return df


def merge_actual_train_test(train, test):
    train['Split'] = 'Train'
    test['Split'] = 'Test'

    df = pd.concat([train, test], axis=0)

    return df


def split_train_test_by_col(df):
    train = df.loc[df['Split'] == 'Train']
    test = df.loc[df['Split'] == 'Test']

    train.drop("Split", axis=1, inplace=True)
    test.drop("Split", axis=1, inplace=True)

    return train, test


def preprocess_data(name_of_dataset, prediction_type):
    if "splitted_data" not in os.listdir():
        os.makedirs("splitted_data")

    if name_of_dataset not in os.listdir("splitted_data/"):
        os.makedirs("splitted_data/"+name_of_dataset)

    if prediction_type not in os.listdir("splitted_data/"+name_of_dataset):
        os.makedirs("splitted_data/"+name_of_dataset+"/"+prediction_type)

    if not "data.csv" in os.listdir("splitted_data/" + name_of_dataset+"/" +
                                    prediction_type):
        # Load dataset
        df = load_dataset(name_of_dataset+"/train.csv")
        val_df = load_dataset(name_of_dataset+"/test.csv")

        if prediction_type == "regression":
            preprocess_obj = WalmartPreprocessingRegression()
            df = preprocess_obj.preprocess(df)
            val_df = preprocess_obj.preprocess(val_df)
            df = move_ycolumn_at_end(df, "Weekly_Sales")
        else:
            preprocess_obj = WalmartPreprocessingForecasting()
            df = merge_actual_train_test(df, val_df)
            df = preprocess_obj.preprocess(df)
            df, val_df = split_train_test_by_col(df)
            val_df.drop("Difference", axis=1, inplace=True)
            df = move_ycolumn_at_end(df, "Difference")

        # Save preprocessed data
        save_dataframe_to_csv(df, name_of_dataset, prediction_type, "data.csv")
        save_dataframe_to_csv(val_df, name_of_dataset,
                              prediction_type, "val.csv")

    else:
        df = load_dataset("splitted_data/"+name_of_dataset +
                          "/"+prediction_type+"/data.csv")

    return df


def get_scaled_data(X_train, X_test, prediction_type):
    # For regression
    # ['IsHoliday' 'Size' 'Temperature' 'Fuel_Price' 'MarkDown1' 'MarkDown2'
    # 'MarkDown3' 'MarkDown4' 'MarkDown5' 'CPI' 'Unemployment' 'md1_present'
    # 'md2_present' 'md3_present' 'md4_present' 'md5_present' Year	Day	Days_Until_Christmas 'Type_A'
    # 'Type_B' 'Store_1' 'Store_2' 'Store_3' 'Dept_1' 'Dept_2' 'Dept_3'
    # 'Dept_4' 'Dept_5' 'Month_1' 'Month_11' 'Month_12' 'Month_2' 'Month_3'
    # 'Month_4' 'Month_5' 'Month_6' 'Month_7' 'Black_Friday_no'
    # 'Pre_christmas_no']

    # For forecasting
    # 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
    #  'MarkDown5', 'CPI', 'Unemployment', 'md1_present', 'md2_present', 'md3_present', 'md4_present',
    #  'md5_present', 'Type_A', 'Type_B', 'Store_1', 'Store_2', 'Store_3', 'Dept_1', 'Dept_2',
    #  'Dept_3', 'Dept_4', 'Dept_5', 'Black_Friday_no', 'Black_Friday_yes', 'Pre_christmas_no',
    #  'Pre_christmas_yes', 'Lagged_Sales', 'Lagged_Available', 'Sales_diff'

    if prediction_type == "regression":
        col_indexes_to_scale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18]
    else:
        col_indexes_to_scale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 32]

    X_train, X_test = scaling.standard_train_test_scaling(
        X_train, X_test, col_indexes_to_scale)

    return X_train, X_test


def train_test_split(df, tdp, dataset, prediction_type):
    if "splitted_data" not in os.listdir():
        os.makedirs("splitted_data")

    if dataset not in os.listdir("splitted_data/"):
        os.makedirs("splitted_data/"+dataset)

    if prediction_type not in os.listdir("splitted_data/"+dataset+"/"):
        os.makedirs("splitted_data/"+dataset+"/"+prediction_type)

    if "train.csv" in os.listdir("splitted_data/"+dataset+"/"+prediction_type+"/"):
        train = pd.read_csv("splitted_data/"+dataset+"/" +
                            prediction_type+"/train.csv")
        test = pd.read_csv("splitted_data/"+dataset+"/" +
                           prediction_type+"/test.csv")
    else:
        df = pd.read_csv("splitted_data/"+dataset+"/" +
                         prediction_type+"/data.csv")
        # Drop columns with all null values
        df = df.dropna(axis=1, how='all')

        # train-test split
        np.random.seed(42)
        train, test = model_selection.train_test_split(
            df, test_size=tdp/100, random_state=42)

        train.to_csv("splitted_data/"+dataset+"/" +
                     prediction_type+"/train.csv", index=False)
        test.to_csv("splitted_data/"+dataset+"/" +
                    prediction_type+"/test.csv", index=False)

    print(train.values.shape, test.values.shape)

    return train.values[:, 0:-1], train.values[:, -1], test.values[:, 0:-1], test.values[:, -1]


def train_gp_regression_model(generation_count, program_list, training_prediction_error, fitness_error_scores,
                              vr_obj, X_train, y_train, gap, dataset, MAX_PROGRAM_SIZE,
                              t, rp, st, NUMBER_OF_REGISTERS):
    # Detection tracking
    # dt_obj = DetectionTracking(list(register_class_map.values()))
    dt_obj = None

    # Attributes of dataset
    source_x = list(range(0, X_train.shape[1] - 1))

    # Registers
    target_r = list(range(0, NUMBER_OF_REGISTERS))
    source_r = list(range(0, NUMBER_OF_REGISTERS))

    # Operators
    # ops = [0, 1, 2, 3]
    ops = [0, 1, 2, 3, 4]

    # Sampling function mapper
    sampling_mapper = {"uniformly": sampling.uniform_sampling}
    sample_flag = False

    # Population size
    actual_population = len(program_list)

    # Best program list
    best_programs = []

    for gen in range(generation_count):
        print("Generation: ", gen+1)

        if (gen % rp) == 0:
            print("Sampling..")
            X_train_t, y_train_t = sampling_mapper[st](X_train, y_train, t)
            sample_flag = True

        # Breeder model for selection-replacement
        fitness_error_scores = selection.get_fitness_error_scores(
            fitness_error_scores, program_list, vr_obj, X_train_t, y_train_t, NUMBER_OF_REGISTERS,
            gen, gap, sample_flag, dt_obj)

        sample_flag = False

        program_list, fitness_error_scores = selection.rank_remove_worst_gap(
            gap, fitness_error_scores, program_list, dataset)

        # Update detection tracker for best individual
        fit_score_best = selection.get_fitness_error_score_of_program(
            program_list[0], X_train_t, y_train_t, NUMBER_OF_REGISTERS, gen, dt_obj, False)

        print("Best fitness score: ", fit_score_best)
        # Training error of fittest program in each generation
        training_prediction_error.append(
            round(list(fitness_error_scores.values())[0], 2))

        # Saving the best program in each generation
        best_programs.append(program_list[0])

        selected_programs = selection.select_for_variation(
            int((gap/100)*actual_population), program_list)

        # Variations

        # Crossover
        offsprings = variation.crossover(
            selected_programs, gap, MAX_PROGRAM_SIZE)

        # Mutation
        offsprings = variation.mutation(
            offsprings, source_x, target_r, source_r, ops, vr_obj, X_train_t, y_train_t)

        # Adding offsprings to program list
        program_list = program_list + offsprings

    return training_prediction_error, best_programs, dt_obj


def train_unsampled_regression_model(generation_count, program_list, training_prediction_error, fitness_error_scores,
                                     vr_obj, X_train, y_train, gap, dataset, MAX_PROGRAM_SIZE,
                                     t, rp, st, NUMBER_OF_REGISTERS):
    dt_obj = None

    # Attributes of dataset
    source_x = list(range(0, X_train.shape[1] - 1))

    # Registers
    target_r = list(range(0, NUMBER_OF_REGISTERS))
    source_r = list(range(0, NUMBER_OF_REGISTERS))

    # Operators
    ops = [0, 1, 2, 3]

    # Sampling
    sample_flag = False

    # Population size
    actual_population = len(program_list)

    # Best program list
    best_programs = []

    for gen in range(generation_count):
        print("Generation: ", gen+1)

        # Breeder model for selection-replacement
        fitness_error_scores = selection.get_fitness_error_scores(
            fitness_error_scores, program_list, vr_obj, X_train, y_train, NUMBER_OF_REGISTERS,
            gen, gap, sample_flag, dt_obj)
        # print(fitness_error_scores)

        program_list, fitness_error_scores = selection.rank_remove_worst_gap(
            gap, fitness_error_scores, program_list, dataset)

        # Update detection tracker for best individual
        fit_score_best = selection.get_fitness_error_score_of_program(
            program_list[0], X_train, y_train, NUMBER_OF_REGISTERS, gen, dt_obj, False)

        # Training error of fittest program in each generation
        training_prediction_error.append(
            round(list(fitness_error_scores.values())[0], 2))

        # Saving the best program in each generation
        best_programs.append(program_list[0])

        selected_programs = selection.select_for_variation(
            int((gap/100)*actual_population), program_list)

        # Variations

        # Crossover
        offsprings = variation.crossover(
            selected_programs, gap, MAX_PROGRAM_SIZE)

        # Mutation
        offsprings = variation.mutation(
            offsprings, source_x, target_r, source_r, ops, vr_obj, X_train, y_train)

        # Adding offsprings to program list
        program_list = program_list + offsprings

    return training_prediction_error, best_programs, dt_obj


def save_and_test_champ_predictor(training_prediction_error, program_list, X_train, y_train, X_test, y_test,
                                  NUMBER_OF_REGISTERS, dataset, st, prediction_type):
    # Check and save best classifier - training data
    prediction.predict_and_save_best_program(training_prediction_error, program_list, X_train, y_train, X_test, y_test,
                                             prediction_type, NUMBER_OF_REGISTERS, dataset, st)

    # Check saved model accuracy - test data
    # prediction_list = prediction.predict(
    #     X_test, "best_programs/"+dataset+"/"+prediction_type+"/MAE/best_program.json", NUMBER_OF_REGISTERS)

    # prediction.save_confusion_matrix(
    #     y_test, prediction_list1, dataset, st, "accuracy")
