# coding: utf-8

# std
import json
import time
import timeit
import random
import argparse
import traceback

# Third party
import numpy as np
import pandas as pd

# application imports
import operations
import population
import prediction
import generation
from variable_references import VariableReference

__author__ = "Aamir Shoeb Alam Khan"
__email__ = "am754815@dal.ca"

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Canonical GP Classifier')
    p.add_argument('--tdp', metavar='test_data_proportion', type=int,
                   help='Test data porportion (value between 10-30) for train-test split, default to 20',
                   choices=[10, 15, 20, 25, 30], default=20)
    p.add_argument('--gap', metavar='gap_percent', type=int,
                   help='Gap percentage for breeder model (value between 10-30), default to 20',
                   choices=[10, 15, 20, 25, 30], default=20)
    p.add_argument('--d', metavar='dataset', type=str,
                   help='Dataset to be used, default to iris', choices=["iris", "tic-tac-toe"], default="iris")
    p.add_argument('--p', metavar='population_size', type=int,
                   help='Size of population, default to 100', choices=[100, 500, 1000, 5000, 10000], default=100)
    p.add_argument('--g', metavar='Number of generations', type=int,
                   help='Number of generations, default to 100', choices=[100, 500, 1000], default=100)
    p.add_argument('--nr', metavar='Number of registers', type=int,
                   help='Number of registers, default to 4', default=4)

    args = p.parse_args()

    print('Parameters:')
    print("""\tTest data porportion: %d\n\tGap percentage: %d\n\tPopulation size: %d\n\tNumber of Generations: %d
        Dataset: %s""" % (args.tdp, args.gap, args.p, args.g, args.d))

    # OTHER PARAMETERS
    MAX_PROGRAM_SIZE = 64
    NUMBER_OF_REGISTERS = args.nr

    try:
        overall_start = timeit.default_timer()

        operators_mapping = {0: operations.add, 1: operations.sub,
                             2: operations.mul_by_2, 3: operations.div_by_2}

        vr_obj = VariableReference(NUMBER_OF_REGISTERS)
        print('\nInitial Registers:')
        print(vr_obj.get_registers())

        # Load dataset
        df = pd.read_csv(args.d+"/"+args.d+".data", header=None)

        # One-hot-encode attributes if tic-tac-toe dataset
        if args.d == "tic-tac-toe":
            df = pd.get_dummies(df, columns=list(
                df.columns.values)[:-1], drop_first=True)
            # Removing & appending label column at the end and renaming the columns
            col_9 = df.pop(9)
            df[9] = col_9
            df.columns = list(range(len(df.columns.values)))

        # Initialize population: returns list of individuals
        program_list = population.initialize_population(
            df, operators_mapping, args.p)

        # Run GP
        generation.run_each_generation(
            program_list, args.g, df, args.gap, args.tdp, MAX_PROGRAM_SIZE, args.d, NUMBER_OF_REGISTERS, vr_obj)

        print('Overall time: '+str(round(timeit.default_timer() -
                                         overall_start, 3))+' seconds.\n')

    except Exception as err:
        print("Unexpected problem encountered: \n", str(err))
        traceback.print_exc()
