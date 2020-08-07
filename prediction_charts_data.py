import prediction
import pandas as pd


def save_data_for_forecast(df):
    df_store_1 = df.loc[df["Store_1"] == 1]
    last_100_rows = df_store_1.tail(100)
    last_100_rows.to_csv("forecast_test.csv", index=False)
    return


def predict_and_print_values(dataset, NUMBER_OF_REGISTERS):
    df = pd.read_csv("forecast_test_scaled.csv", header=None)
    data = df.values
    X_val = data[:, 0:-1]
    y_val = data[:, -1]
    prediction.print_ytest_ypred(
        y_val, X_val, dataset, "forecast", NUMBER_OF_REGISTERS)
