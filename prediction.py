import os
import ast
import json
from pycm import *
import numpy as np
from sklearn import metrics

import selection
import operations
from variable_references import VariableReference


def load_best_program(program_path):
    if os.path.exists(program_path):
        with open(program_path, "r") as json_file:
            data = json.load(json_file)

        program = ast.literal_eval(data["program"])

        return program

    return []


def get_mae_prediction_error(y_pred, y_actual):
    mae = metrics.mean_absolute_error(y_actual, y_pred)

    return mae


def predictor_accuracy(y_pred, y_actual):
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    if type(y_actual) == list:
        y_actual = np.array(y_actual)

    tp = sum(y_actual == y_pred)
    accuracy = tp/len(y_pred)

    print("Accuracy: ", accuracy)

    # pred_score = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] == y_actual[i]:
    #         pred_score += 1

    # return pred_score/len(y_pred)

    return accuracy


# def classifier_detection_rate(y_pred, y_actual, number_of_labels):
#     sum_dr = 0.0

#     if type(y_pred) == list:
#         y_pred = np.array(y_pred)

#     if type(y_actual) == list:
#         y_actual = np.array(y_actual)

#     unique_labels = set(y_actual.tolist())
#     print(y_actual, y_pred, unique_labels)
#     # Sum the detection rate for each label
#     for label in unique_labels:
#         tp = sum((y_actual == label) & (y_pred == label))
#         fn = sum((y_actual == label) & (y_pred != label))

#         dr = tp / (tp + fn)
#         # print("DR for label, ", label, ": ", dr)
#         sum_dr += dr

#     DR = sum_dr / number_of_labels
#     print("Avg DR: ", DR)

#     return DR


def predict(input_list, program_path, NUMBER_OF_REGISTERS):
    prediction_list = []

    if type(program_path) == str:
        program = load_best_program(program_path)
    else:
        program = program_path

    if program:
        vr_obj = VariableReference(NUMBER_OF_REGISTERS)

        # operators_mapping = {0: operations.add, 1: operations.sub,
        #                      2: operations.mul_by_2, 3: operations.div_by_2}
        operators_mapping = {0: operations.add, 1: operations.sub,
                             2: operations.mul_by_2, 3: operations.div_by_2,
                             4: operations.conditional}

        for idx in range(len(input_list)):
            # Reset and get registers
            vr_obj.reset_registers()
            registers = vr_obj.get_registers()

            for instruction in program:
                # Decode and Execute
                registers = selection.decode_execute_instruction(
                    input_list[idx], instruction, operators_mapping, registers)

            sorted_registers = {k: v for k, v in sorted(
                registers.items(), key=lambda item: item[1], reverse=True)}

            # for k in sorted_registers:
            #     if k in register_class_map.keys():
            #         prediction_list.append(register_class_map[k])
            #         break
            prediction_list.append(list(sorted_registers.values())[0])
    else:
        print("No saved program found")

    return prediction_list


def save_champion_predictor(program, score_type, train_score, test_score, avg_score, dataset, prediction_type):
    data = {}

    if "best_programs" not in os.listdir():
        os.makedirs("best_programs")

    if dataset not in os.listdir("best_programs/"):
        os.makedirs("best_programs/"+dataset)

    if prediction_type not in os.listdir("best_programs/"+dataset+"/"):
        os.makedirs("best_programs/"+dataset+"/"+prediction_type)

    if "best_program.json" in os.listdir("best_programs/"+dataset+"/"+prediction_type+"/"):
        with open("best_programs/"+dataset+"/"+prediction_type+"/best_program.json", "r") as json_file:
            data = json.load(json_file)

    if data:
        if avg_score < data["Avg_"+score_type]:
            data["Avg_"+score_type] = avg_score
            data["Train_"+score_type] = train_score
            data["Test_"+score_type] = test_score
            data["program"] = str(program)

            with open("best_programs/"+dataset+"/"+prediction_type+"/best_program.json", 'w') as outfile:
                json.dump(data, outfile)
    else:
        data["Avg_"+score_type] = avg_score
        data["Train_"+score_type] = train_score
        data["Test_"+score_type] = test_score
        data["program"] = str(program)

        with open("best_programs/"+dataset+"/"+prediction_type+"/best_program.json", 'w') as outfile:
            json.dump(data, outfile)


def get_binary_outcome(pred_list):
    pred_list_binary = []

    for pred in pred_list:
        if pred < 0:
            pred_list_binary.append(0)
        else:
            pred_list_binary.append(1)

    return pred_list_binary


def get_binary_prediction_lists(y_test, X_test, dataset, NUMBER_OF_REGISTERS):
    program_path = "best_programs/"+dataset+"/forecast/best_program.json"
    best_program = load_best_program(program_path)

    y_pred = predict(X_test, best_program, NUMBER_OF_REGISTERS)

    y_test_binary = get_binary_outcome(y_test)
    y_pred_binary = get_binary_outcome(y_pred)

    return y_pred_binary, y_test_binary


def print_ytest_ypred(y_test, X_test, dataset, pt, NUMBER_OF_REGISTERS):
    program_path = "best_programs/"+dataset+"/"+pt+"/best_program.json"
    best_program = load_best_program(program_path)

    y_pred = predict(X_test, best_program, NUMBER_OF_REGISTERS)
    print(len(y_pred), len(y_test))
    print("ytest: ", y_test.tolist())
    print("ypred: ", y_pred)


def predict_and_save_best_program(training_prediction_error, best_programs, X_train, y_train, X_test, y_test,
                                  prediction_type, NUMBER_OF_REGISTERS, dataset, st):
    test_prediction_error = []
    avg_train_test_error = []
    # y_train_binary = get_binary_outcome(y_test)

    for program in best_programs:
        y_pred = predict(X_test, program, NUMBER_OF_REGISTERS)

        program_err = get_mae_prediction_error(y_pred, y_test)

        test_prediction_error.append(program_err)

    best_prog = best_programs[0]
    best_mae = (training_prediction_error[0] + test_prediction_error[0])/2
    mae_training = training_prediction_error[0]
    mae_test = test_prediction_error[0]

    print("Training error: ", training_prediction_error)
    print("Testing error: ", test_prediction_error)

    for idx in range(1, len(best_programs)):
        avg_error = (
            training_prediction_error[idx] + test_prediction_error[idx])/2

        avg_train_test_error.append(avg_error)

        if avg_error < best_mae:
            best_mae = avg_error
            best_prog = best_programs[idx]
            mae_training = training_prediction_error[idx]
            mae_test = test_prediction_error[idx]

    # Save predictor with least Mean Absolute Error
    save_champion_predictor(best_prog, "MAE", mae_training, mae_test, best_mae,
                            dataset, prediction_type)

    # Predict, convert to binary, and save confusion matrix for forecast
    if prediction_type == "forecast":
        y_pred_binary, y_test_binary = get_binary_prediction_lists(
            y_test, X_test, dataset, NUMBER_OF_REGISTERS)

        save_confusion_matrix(y_test_binary, y_pred_binary,
                              dataset, prediction_type)
    else:
        print_ytest_ypred(y_test, X_test, dataset,
                          prediction_type, NUMBER_OF_REGISTERS)


def save_confusion_matrix(y_true, y_pred, dataset, pt):
    filename = 'confusion_matrix_'+dataset+'_'+pt

    cm = ConfusionMatrix(y_true, y_pred)
    cm.save_html(filename, color=(100, 50, 250))
