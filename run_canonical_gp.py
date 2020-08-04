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
import walmart
import rossmann
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
                   help='Dataset to be used, default to walmart', choices=["walmart", "rossmann"],
                   default="walmart")
    p.add_argument('--p', metavar='population_size', type=int,
                   help='Size of population, default to 100', choices=[100, 500, 1000, 5000, 10000], default=100)
    p.add_argument('--g', metavar='Number of generations', type=int,
                   help='Number of generations, default to 1000', choices=[100, 500, 1000, 5000], default=100)
    p.add_argument('--nr', metavar='Number of registers', type=int,
                   help='Number of registers, default to 4', default=4)
    p.add_argument('--t', metavar='Training subset size', type=int,
                   help='Training subset size, default to 200', choices=[200, 300, 500], default=200)
    p.add_argument('--st', metavar='Sampling technique', type=str,
                   help="Sampling technique, default to 'uniformly'", choices=["uniformly"],
                   default="uniformly")
    p.add_argument('--rp', metavar='Re-sampling period', type=int,
                   help='Re-sampling period, default to 5 generations', choices=[5, 10, 15, 20, 25], default=5)
    p.add_argument('--pt', metavar='Prediction technique', type=str,
                   help="Prediction technique, default to 'forecast'", choices=["forecast", "regression"],
                   default="forecast")

    args = p.parse_args()

    print('Parameters:')
    print("""\tTest data porportion: %d\n\tGap percentage: %d\n\tPopulation size: %d\n\tNumber of Generations: %d
        Dataset: %s\n\tSampling technique: %s\n\tPrediction technique: %s\n\tTraining subset size: %d\n\tResampling period: %d"""
          % (args.tdp, args.gap, args.p, args.g, args.d, args.st, args.pt, args.t, args.rp))

    # OTHER PARAMETERS
    MAX_PROGRAM_SIZE = 64
    NUMBER_OF_REGISTERS = args.nr

    try:
        overall_start = timeit.default_timer()

        operators_mapping = {0: operations.add, 1: operations.sub,
                             2: operations.mul_by_2, 3: operations.div_by_2}

        data_module_mapping = {"walmart": walmart, "rossmann": rossmann}

        # Validate and get number of registers
        NUMBER_OF_REGISTERS = data_module_mapping[args.d].validate_get_register_count(
            NUMBER_OF_REGISTERS)

        vr_obj = VariableReference(NUMBER_OF_REGISTERS)
        print('\nInitial Registers:')
        print(vr_obj.get_registers())

        # Preprocess
        df = data_module_mapping[args.d].preprocess_data(args.d, args.pt)

        # Initialize population: returns list of individuals
        program_list = population.initialize_population(
            df, operators_mapping, args.p, NUMBER_OF_REGISTERS)

        # Run GP
        # generation.run_each_generation(
        #     program_list, args.g, df, args.gap, args.tdp, MAX_PROGRAM_SIZE, args.d,
        #     NUMBER_OF_REGISTERS, vr_obj, data_module_mapping, args.t, args.rp, args.st,
        #     args.pt)

        print('Overall time: '+str(round(timeit.default_timer() -
                                         overall_start, 3))+' seconds.\n')

    except Exception as err:
        print("Unexpected problem encountered: \n", str(err))
        traceback.print_exc()
