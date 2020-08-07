import os
import json
import selection
import variation
import prediction
import prediction_charts_data


def run_each_generation(program_list, generation_count, df, gap, tdp, MAX_PROGRAM_SIZE,
                        dataset, NUMBER_OF_REGISTERS, vr_obj, data_module_mapping, t, rp, st, pt):
    X_train, y_train, X_test, y_test = data_module_mapping[dataset].train_test_split(
        df, tdp, dataset, pt)

    # Scaling(if required)
    X_train, X_test = data_module_mapping[dataset].get_scaled_data(
        X_train, X_test, pt)

    training_prediction_error = []
    fitness_scores = {}

    # Put training function here
    training_prediction_error, best_programs, dt_obj = data_module_mapping[dataset].train_gp_regression_model(generation_count, program_list, training_prediction_error, fitness_scores,
                                                                                                              vr_obj, X_train, y_train, gap, dataset, MAX_PROGRAM_SIZE,
                                                                                                              t, rp, st, NUMBER_OF_REGISTERS
                                                                                                              )

    # # Check, save and print metrics of best classifier
    data_module_mapping[dataset].save_and_test_champ_predictor(training_prediction_error, best_programs, X_train, y_train,
                                                               X_test, y_test, NUMBER_OF_REGISTERS, dataset, st, pt)

    # Prediction results
    # if pt == "regression":
    #     prediction.print_ytest_ypred(
    #         y_test, X_test, dataset, NUMBER_OF_REGISTERS)
    # else:
    #     prediction_charts_data.predict_and_print_values(
    #         dataset, NUMBER_OF_REGISTERS)

    # # Dumping detection tracking json object to a file
    # with open('dt_'+dataset+'_'+st+'.json', 'w') as outfile:
    #     json.dump(dt_obj.get_label_wise_tracker(), outfile)
