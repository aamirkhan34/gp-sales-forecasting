import random
import numpy as np
import pandas as pd


def uniform_random_choice(choice_list):
    # Chooses a value from a list uniformly
    return random.choice(choice_list)


def get_program_counter(pop_size):
    # Chooses program size for population uniformly
    uniform_pc_array = np.random.uniform(15, 45, pop_size)
    return [int(round(x, 0)) for x in list(uniform_pc_array)]


def get_individuals(pc_list, select, target_r, source_r, ops, source_x):
    prog_list = []

    for prog_size in pc_list:
        # print("PROGRAM SIZE: ", prog_size)
        prog = []
        for i in range(prog_size):
            instruction = []

            select_choice = uniform_random_choice(select)
            if select_choice:
                instruction.append(select_choice)
                instruction.append(uniform_random_choice(target_r))
                instruction.append(uniform_random_choice(ops))
                instruction.append(uniform_random_choice(source_r))

            else:
                instruction.append(select_choice)
                instruction.append(uniform_random_choice(target_r))
                instruction.append(uniform_random_choice(ops))
                instruction.append(uniform_random_choice(source_x))

            prog.append(instruction)

        prog_list.append(prog)

    return prog_list


def initialize_population(df, operators_mapping, pop_size, NUMBER_OF_REGISTERS):
    # Select bit
    select = [0, 1]

    # Attributes of dataset
    source_x = list(range(0, len(df.columns.values[:-1])))

    # Registers
    target_r = list(range(0, NUMBER_OF_REGISTERS))
    source_r = list(range(0, NUMBER_OF_REGISTERS))

    # Operators
    ops = list(operators_mapping.keys())

    # print(select, target_r, ops, source_r, source_x)

    pc_list = get_program_counter(pop_size)

    program_list = get_individuals(
        pc_list, select, target_r, source_r, ops, source_x)

    return program_list
