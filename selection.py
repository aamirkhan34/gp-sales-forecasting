import os
import json
import random
import operations
import prediction
import numpy as np
import pandas as pd


def decode_execute_instruction(row, instruction, operators_mapping, registers):
    # Decode
    target = instruction[1]
    operator = instruction[2]
    source = instruction[3]

    # Execute
    if instruction[0]:
        # Register
        registers["R"+str(target)] = operators_mapping[operator](
            registers["R"+str(target)], registers["R"+str(source)])
    else:
        # Input Data
        registers["R"+str(target)] = operators_mapping[operator](
            registers["R"+str(target)], row[source])

    return registers


def get_fitness_error_score_of_program(program, X_train, y_train, NUMBER_OF_REGISTERS, gen, dt_obj, det_track_req):
    operators_mapping = {0: operations.add, 1: operations.sub,
                         2: operations.mul_by_2, 3: operations.div_by_2}

    # Predict
    y_pred = prediction.predict(X_train, program, NUMBER_OF_REGISTERS)

    fitness_error_score = prediction.get_mae_prediction_error(y_pred, y_train)

    # for idx in range(len(X_train)):
    #     # Reset and get registers
    #     vr_obj.reset_registers()
    #     registers = vr_obj.get_registers()

    #     for instruction in program:
    #         # Decode and Execute
    #         registers = decode_execute_instruction(
    #             X_train[idx], instruction, operators_mapping, registers)

    #     # y_pred =

    #     # # Update detection tracking if required - total
    #     # if det_track_req:
    #     #     dt_obj.set_label_wise_tracker_total(y_train[idx], gen)

    #     sum_fitness_error_score += pred_error_score
    #     # # Update detection tracking if required - true positive
    #     # if det_track_req:
    #     #     dt_obj.set_label_wise_tracker_tp(
    #     #         y_train[idx], gen, match_check)

    return fitness_error_score


def get_fitness_error_scores(fitness_error_scores, program_list, vr_obj, X_train, y_train, NUMBER_OF_REGISTERS,
                             gen, gap, sample_flag, dt_obj):
    operators_mapping = {0: operations.add, 1: operations.sub,
                         2: operations.mul_by_2, 3: operations.div_by_2}

    # Change range if
    if gen > 0 and not sample_flag:
        rg = range(int(len(program_list)*(1-gap/100)),
                   len(program_list))
    else:
        rg = range(len(program_list))

    for pidx in rg:

        fitness_error_score = get_fitness_error_score_of_program(
            program_list[pidx], X_train, y_train, NUMBER_OF_REGISTERS, gen, dt_obj, False)

        # for idx in range(len(X_train)):
        #     # Reset and get registers
        #     vr_obj.reset_registers()
        #     registers = vr_obj.get_registers()

        #     for instruction in program_list[pidx]:
        #         # Decode and Execute
        #         registers = decode_execute_instruction(
        #             X_train[idx], instruction, operators_mapping, registers)

        #     match_check = check_classification(
        #         registers, register_class_map, y_train[idx])

        #     if match_check:
        #         fitness_score += 1

        fitness_error_scores[pidx] = fitness_error_score

    return fitness_error_scores


def save_fittest_individual(program, fitness_error_score, dataset):
    data = {}

    if "fittest_programs" not in os.listdir():
        os.makedirs("fittest_programs")

    if dataset not in os.listdir("fittest_programs/"):
        os.makedirs("fittest_programs/"+dataset)

    if "fittest_program.json" in os.listdir("fittest_programs/"+dataset+"/"):
        with open("fittest_programs/"+dataset+"/fittest_program.json", "r") as json_file:
            data = json.load(json_file)

    if data:
        if data["fitness_error_score"] > fitness_error_score:
            data["fitness_error_score"] = fitness_error_score
            data["program"] = str(program)
            # data["label_mapping"] = str(register_class_map)

            with open("fittest_programs/"+dataset+"/fittest_program.json", 'w') as outfile:
                json.dump(data, outfile)
    else:
        data["fitness_error_score"] = fitness_error_score
        data["program"] = str(program)
        # data["label_mapping"] = str(register_class_map)

        with open("fittest_programs/"+dataset+"/fittest_program.json", 'w') as outfile:
            json.dump(data, outfile)


def rank_remove_worst_gap(gap, fitness_error_scores, program_list, dataset):
    # Ranking by fitness error scores: Less is better
    sorted_fitness_error_scores = {k: v for k, v in sorted(
        fitness_error_scores.items(), key=lambda item: item[1])}
    print("Sorted: ", list(sorted_fitness_error_scores.items())[:3])

    sorted_program_list = [program_list[idx]
                           for idx, score in sorted_fitness_error_scores.items()]

    # Remove weak programs
    program_list = sorted_program_list[:int(
        len(sorted_program_list)*(1-gap/100))]

    # Save fittest program if fittest
    save_fittest_individual(
        program_list[0], fitness_error_scores[list(fitness_error_scores.keys())[0]], dataset)

    print(len(program_list))

    # Resetting the indices of sorted_fitness_error_scores to avoid repeated fitness evaluation
    new_fitness_map = {idx: score for idx, score in enumerate(
        list(sorted_fitness_error_scores.values())[:int(len(sorted_program_list)*(1-gap/100))])}

    return program_list, new_fitness_map


def select_for_variation(parents_size, program_list):
    # Selection of parents with uniform distribution
    selected_programs = random.choices(program_list, k=parents_size)

    return selected_programs
