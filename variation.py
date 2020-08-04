import copy
import random
import numpy as np


import selection
import population


def valid_prog_size(program, repl_points, ins_points, MAX_PROGRAM_SIZE):
    program_size = len(
        program) - (repl_points[1] - repl_points[0]) + (ins_points[1] - ins_points[0])

    if program_size > MAX_PROGRAM_SIZE:
        return False

    return True


def two_point_crossover(prog1, prog2, MAX_PROGRAM_SIZE):
    prog1_matrix = np.array(prog1)
    prog2_matrix = np.array(prog2)

    min_size = min(len(prog1), len(prog2))

    xover_p1 = random.randint(1, min_size)
    xover_p2 = random.randint(1, min_size - 1)

    if xover_p2 >= xover_p1:
        xover_p2 += 1
    else:
        # Swapping the two xover points
        xover_p1, xover_p2 = xover_p2, xover_p1

    org_prog1_matrix = copy.deepcopy(prog1_matrix)
    prog1_matrix[xover_p1:xover_p2] = prog2_matrix[xover_p1:xover_p2]
    prog2_matrix[xover_p1:xover_p2] = org_prog1_matrix[xover_p1:xover_p2]

    return prog1_matrix.tolist(), prog2_matrix.tolist()


def get_program_replacement_points(prog1, prog2):
    if len(prog1) == 1:
        prog1_points = [0, 1]
    else:
        prog1_points = random.sample(list(range(0, len(prog1))), k=2)

    if len(prog2) == 1:
        prog2_points = [0, 1]
    else:
        prog2_points = random.sample(list(range(0, len(prog2))), k=2)

    prog1_points.sort()
    prog2_points.sort()

    return prog1_points, prog2_points


def insert_xover_elements(data_to_be_inserted, program, insert_index):
    for instruction in data_to_be_inserted:
        program.insert(insert_index, instruction)
        insert_index += 1

    return program


def evolving_two_point_crossover(prog1, prog2, MAX_PROGRAM_SIZE):
    xover_success = False

    prog1_points, prog2_points = get_program_replacement_points(prog1, prog2)

    if valid_prog_size(prog1, prog1_points, prog2_points, MAX_PROGRAM_SIZE) and valid_prog_size(prog2, prog2_points, prog1_points, MAX_PROGRAM_SIZE):
        prog1_replacement_index = prog1_points[0]
        prog2_replacement_index = prog2_points[0]

        prog1_xover_data = prog1[prog1_points[0]:prog1_points[1]]
        prog2_xover_data = prog2[prog2_points[0]:prog2_points[1]]

        # Delete sliced entries
        del prog1[prog1_points[0]:prog1_points[1]]
        del prog2[prog2_points[0]:prog2_points[1]]

        # Perform crossover
        prog1 = insert_xover_elements(
            prog2_xover_data, prog1, prog1_replacement_index)

        prog2 = insert_xover_elements(
            prog1_xover_data, prog2, prog2_replacement_index)

        xover_success = True

    return prog1, prog2, xover_success


def crossover(selected_programs, gap, MAX_PROGRAM_SIZE):
    offsprings = []

    sel_prog_size = len(selected_programs)

    for i in range(int(sel_prog_size/2)):
        offspring1, offspring2, success = evolving_two_point_crossover(
            selected_programs[i], selected_programs[sel_prog_size-1-i], MAX_PROGRAM_SIZE)

        # Perform fixed length two point crossover if evolving crossover fails
        if not success:
            offspring1, offspring2 = two_point_crossover(
                selected_programs[i], selected_programs[sel_prog_size-1-i], MAX_PROGRAM_SIZE)

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return offsprings


def swap_mutation(offspring):
    size = range(len(offspring))
    i, j = random.sample(size, 2)
    # Swapping
    old_i_element = offspring[i]
    offspring[i] = offspring[j]
    offspring[j] = old_i_element

    return offspring


def get_flipped_value(old_value, choice_list):
    new_choice_list = [x for x in choice_list if x != old_value]
    new_value = population.uniform_random_choice(new_choice_list)
    return new_value


def flip_bit_mutation(individual, source_x, target_r, source_r, ops):
    mutated_individual = [[bit for bit in instruction]
                          for instruction in individual]

    flip_choice = ["target", "operator", "source"]
    to_be_flipped = population.uniform_random_choice(flip_choice)

    instruction_indices = list(range(0, len(individual)))
    to_be_mutated = population.uniform_random_choice(instruction_indices)

    # Flipping
    if to_be_flipped == "target":
        cur_target = individual[to_be_mutated][1]
        new_target = get_flipped_value(cur_target, target_r)
        mutated_individual[to_be_mutated][1] = new_target
        # print("target", to_be_mutated, cur_target, new_target)

    elif to_be_flipped == "operator":
        cur_op = individual[to_be_mutated][2]
        new_op = get_flipped_value(cur_op, ops)
        mutated_individual[to_be_mutated][2] = new_op
        # print("operator", to_be_mutated, cur_op, new_op)

    else:
        cur_source = individual[to_be_mutated][3]
        cur_select = individual[to_be_mutated][0]
        if cur_select:
            new_source = get_flipped_value(cur_source, source_r)
        else:
            new_source = get_flipped_value(cur_source, source_x)
        mutated_individual[to_be_mutated][3] = new_source
        # print("source", to_be_mutated, cur_source, new_source)

    # print(mutated_individual)
    return mutated_individual


def compare_fitness_scores(offspring, offspring_mutated, vr_obj, X_train_t, y_train_t, register_class_map):
    # Update detection tracker for best individual
    fit_score_offspring = selection.get_fitness_score_of_program(
        offspring, vr_obj, X_train_t, y_train_t, register_class_map, None, None, False)

    # Update detection tracker for best individual
    fit_score_offspring_mutated = selection.get_fitness_score_of_program(
        offspring_mutated, vr_obj, X_train_t, y_train_t, register_class_map, None, None, False)

    return fit_score_offspring == fit_score_offspring_mutated


def mutation(offsprings, source_x, target_r, source_r, ops, vr_obj, X_train_t, y_train_t):
    mutated_offsprings = []

    for offspring in offsprings:
        # mutated_offspring = swap_mutation(offspring)
        mutated_offspring = flip_bit_mutation(
            offspring, source_x, target_r, source_r, ops)

        # Keep on mutating until fitness score is different from original offspring
        # while(compare_fitness_scores(offspring, mutated_offspring, vr_obj, X_train_t,
        #                              y_train_t, register_class_map)):
        #     mutated_offspring = flip_bit_mutation(
        #         mutated_offspring, source_x, target_r, source_r, ops)

        mutated_offsprings.append(mutated_offspring)

    return mutated_offsprings
