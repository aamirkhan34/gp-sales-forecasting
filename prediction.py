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

        operators_mapping = {0: operations.add, 1: operations.sub,
                             2: operations.mul_by_2, 3: operations.div_by_2}

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


def save_champion_classifier(program, score_type, score, register_class_map, dataset, st):
    data = {}

    if "best_programs" not in os.listdir():
        os.makedirs("best_programs")

    if dataset not in os.listdir("best_programs/"):
        os.makedirs("best_programs/"+dataset)

    if st not in os.listdir("best_programs/"+dataset+"/"):
        os.makedirs("best_programs/"+dataset+"/"+st)

    if score_type not in os.listdir("best_programs/"+dataset+"/"+st+"/"):
        os.makedirs("best_programs/"+dataset+"/"+st+"/"+score_type)

    if "best_program.json" in os.listdir("best_programs/"+dataset+"/"+st+"/"+score_type+"/"):
        with open("best_programs/"+dataset+"/"+st+"/"+score_type+"/best_program.json", "r") as json_file:
            data = json.load(json_file)

    if data:
        if data[score_type] < score:
            data[score_type] = score
            data["program"] = str(program)
            data["label_mapping"] = str(register_class_map)

            with open("best_programs/"+dataset+"/"+st+"/"+score_type+"/best_program.json", 'w') as outfile:
                json.dump(data, outfile)
    else:
        data[score_type] = score
        data["program"] = str(program)
        data["label_mapping"] = str(register_class_map)

        with open("best_programs/"+dataset+"/"+st+"/"+score_type+"/best_program.json", 'w') as outfile:
            json.dump(data, outfile)


def predict_and_save_best_classifier(program_list, X_train, y_train, register_class_map,
                                     NUMBER_OF_REGISTERS, dataset, st):
    number_of_labels = len(register_class_map.keys())

    champ_classifier_by_dr = program_list[0]
    champ_classifier_by_acc = program_list[0]

    best_y_pred = predict(
        X_train, program_list[0], NUMBER_OF_REGISTERS, register_class_map)

    best_y_pred_acc = best_y_pred
    best_y_pred_dr = best_y_pred

    champ_classifier_dr = classifier_detection_rate(
        best_y_pred, y_train, number_of_labels)

    champ_classifier_acc = classifier_accuracy(best_y_pred, y_train)

    for program in program_list[1:]:
        y_pred = predict(
            X_train, program, NUMBER_OF_REGISTERS, register_class_map)

        program_accuracy = classifier_accuracy(y_pred, y_train)
        program_detection_rate = classifier_detection_rate(
            y_pred, y_train, number_of_labels)

        if program_accuracy > champ_classifier_acc:
            champ_classifier_acc = program_accuracy
            champ_classifier_by_acc = program
            best_y_pred_acc = y_pred

        if program_detection_rate > champ_classifier_dr:
            champ_classifier_dr = program_detection_rate
            champ_classifier_by_dr = program
            best_y_pred_dr = y_pred

    # Save classifier with highest DR
    # save_champion_classifier(champ_classifier_by_dr, "DetectionRate", champ_classifier_dr,
    #                          register_class_map, dataset, st)

    # Save classifier with highest Accuracy
    save_champion_classifier(champ_classifier_by_acc, "MAE", champ_classifier_acc,
                             register_class_map, dataset, st)

    print(metrics.classification_report(y_train, best_y_pred_acc, digits=3))
    print(metrics.classification_report(y_train, best_y_pred_dr, digits=3))


def save_confusion_matrix(y_true, y_pred, dataset, st, score_type):
    filename = 'confusion_matrix_'+dataset+'_'+st+'_'+score_type

    cm = ConfusionMatrix(y_true, y_pred)
    cm.save_html(filename, color=(100, 50, 250))
