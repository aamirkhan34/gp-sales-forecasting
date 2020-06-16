import os
import ast
import json

import selection
import operations
from variable_references import VariableReference


def load_best_program(program_path):
    if os.path.exists(program_path):
        with open(program_path, "r") as json_file:
            data = json.load(json_file)

        program = ast.literal_eval(data["program"])
        register_class_map = ast.literal_eval(data["label_mapping"])

        return program, register_class_map

    return [], {}


def classifier_accuracy(y_pred, y_actual):
    # print((y_pred == y_actual).all(axis=(0, 2)))
    pred_score = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_actual[i]:
            pred_score += 1

    return pred_score/len(y_pred)
    # return (y_pred == y_actual).all(axis=(0, 2)).mean()


def predict(input_list, program_path, NUMBER_OF_REGISTERS):
    prediction_list = []

    program, register_class_map = load_best_program(program_path)

    if program and register_class_map:
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

            for k in sorted_registers:
                if k in register_class_map.keys():
                    prediction_list.append(register_class_map[k])
                    break
    else:
        print("No saved program found")

    return prediction_list
