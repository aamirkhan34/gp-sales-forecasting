import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


class WalmartPreprocessing(object):

    def preprocess(self, df):
        return df

    def check_update_markdown(self, df):
        # Basic feature engineering by creating a feature which indicates whether a certain markdown was active at all
        df = df.assign(md1_present=df.MarkDown1.notnull())
        df = df.assign(md2_present=df.MarkDown2.notnull())
        df = df.assign(md3_present=df.MarkDown3.notnull())
        df = df.assign(md4_present=df.MarkDown4.notnull())
        df = df.assign(md5_present=df.MarkDown5.notnull())

        # Fill in missing Markdown values
        df["MarkDown1"].fillna(0, inplace=True)
        df["MarkDown2"].fillna(0, inplace=True)
        df["MarkDown3"].fillna(0, inplace=True)
        df["MarkDown4"].fillna(0, inplace=True)
        df["MarkDown5"].fillna(0, inplace=True)

        return df

    def check_update_CPI(self, df):
        df['CPI'] = df['CPI'].fillna(df['CPI'].mean())

        return df

    def check_update_unemployment(self, df):
        df['Unemployment'] = df['Unemployment'].fillna(
            df['Unemployment'].mean())

        return df

    def fill_missing_values(self, df):
        df = self.check_update_markdown(df)
        df = self.check_update_CPI(df)
        df = self.check_update_unemployment(df)

        return df

    def get_one_hot_encoded_types(self, df):
        df['Type'] = 'Type_' + df['Type'].map(str)

        tp = pd.get_dummies(df["Type"])
        df = pd.concat([df, tp], axis=1)

        return df

    def get_one_hot_encoded_stores(self, df):
        df['Store'] = 'Store_' + df['Store'].map(str)

        tp = pd.get_dummies(df["Store"])
        df = pd.concat([df, tp], axis=1)

        return df

    def get_one_hot_encoded_departments(self, df):
        df['Dept'] = 'Dept_' + df['Dept'].map(str)

        tp = pd.get_dummies(df["Dept"])
        df = pd.concat([df, tp], axis=1)

        return df

    def get_encoded_features(self, df):
        df = self.get_one_hot_encoded_types(df)
        df = self.get_one_hot_encoded_stores(df)
        df = self.get_one_hot_encoded_departments(df)

        return df


class WalmartPreprocessingRegression(WalmartPreprocessing):

    def add_year_feature(self, df):
        # Add column for year
        df["Year"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.year

        return df

    # def add_month_feature(self, df):
    #     return df

    def add_day_feature(self, df):
        # Add column for day
        df["Day"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.day

        return df

    def add_days_until_christmas_feature(self, df):
        # Add column for days to next Christmas
        df["Days_Until_Christmas"] = (pd.to_datetime(df["Year"].astype(str)+"-12-31", format="%Y-%m-%d") -
                                      pd.to_datetime(df["Date"], format="%Y-%m-%d")).dt.days.astype(int)

        return df

    def get_date_features(self, df):
        # Sorting data with respect to date since time-series data
        df = df.sort_values(by='Date')

        df = self.add_year_feature(df)
        df = self.add_day_feature(df)
        df = self.add_days_until_christmas_feature(df)

        # Remove Date column
        df.drop("Date", axis=1, inplace=True)

        return df

    def replace_boolean_with_int(self, df):
        df = df.replace({True: 1, False: 0})

        return df

    def drop_encoded_columns(self, df, columns):
        for col in columns:
            df = df.drop(columns=col)

        return df

    def preprocess(self, df):
        df = super().fill_missing_values(df)

        df = self.get_date_features(df)

        df = super().get_encoded_features(df)

        df = self.drop_encoded_columns(df, ["Type", "Store", "Dept"])

        df = self.replace_boolean_with_int(df)

        return df


class WalmartPreprocessingForecasting(WalmartPreprocessing):

    def add_month_feature(self, df):
        # Add column for day
        df["Month"] = pd.to_datetime(
            df["Date_Type"], format="%Y-%m-%d").dt.month

        return df

    def add_black_friday_features(self, df):
        # Add column for days to next Christmas
        df['Black_Friday'] = np.where((df['Date_Type'] == datetime(2010, 11, 26).date()) | (
            df['Date_Type'] == datetime(2011, 11, 25).date()), 'yes', 'no')

        return df

    def add_pre_christmas_features(self, df):
        # Add column for days to next Christmas
        df['Pre_christmas'] = np.where((df['Date_Type'] == datetime(2010, 12, 23).date()) | (df['Date_Type'] == datetime(2010, 12, 24).date()) | (
            df['Date_Type'] == datetime(2011, 12, 23).date()) | (df['Date_Type'] == datetime(2011, 12, 24).date()), 'yes', 'no')

        return df

    def get_date_features(self, df):
        # Convert Date format
        df['Date_Type'] = [datetime.strptime(
            date, '%Y-%m-%d').date() for date in df['Date'].astype(str).values.tolist()]

        # Sorting data with respect to date since time-series data
        # df = df.sort_values(by='Date')

        df = self.add_month_feature(df)
        df = self.add_black_friday_features(df)
        df = self.add_pre_christmas_features(df)
        # df = self.add_day_feature(df)
        # df = self.add_days_until_christmas_feature(df)

        # Remove Date column
        # df.drop("Date", axis=1, inplace=True)

        return df

    def replace_boolean_with_int(self, df):
        df = df.replace({True: 1, False: 0})

        return df

    def get_one_hot_encoded_feature(self, df, column):
        df[column] = column+'_' + df[column].map(str)

        tp = pd.get_dummies(df[column])
        df = pd.concat([df, tp], axis=1)
        # df = df.drop(columns=column)

        return df

    def encode_other_features(self, df):
        # df = self.get_one_hot_encoded_feature(df, "Month")
        df = self.get_one_hot_encoded_feature(df, "Black_Friday")
        df = self.get_one_hot_encoded_feature(df, "Pre_christmas")

        return df

    def get_median_sales(self, df):
        medians = pd.DataFrame({'Median Sales': df.loc[df['Split'] == 'Train'].groupby(
            by=['Type', 'Dept', 'Store', 'Month', 'IsHoliday_x'])['Weekly_Sales'].median()}).reset_index()

        # Merge by type, store, department and month
        df = df.merge(medians, how='outer', on=[
                      'Type', 'Dept', 'Store', 'Month', 'IsHoliday_x'])

        # Fill Null values
        df['Median Sales'].fillna(
            df['Median Sales'].loc[df['Split'] == 'Train'].median(), inplace=True)

        # Create a key for easy access
        df['Key'] = df['Type'].map(
            str)+df['Dept'].map(str)+df['Store'].map(str)+df['Date'].map(str)+df['IsHoliday_x'].map(str)

        return df

    def get_lagged_sales(self, df, sorted_df):
        # Loop over df rows and check at each step if the previous week's sales are available.
        # If not, fill with store and department average, which we retrieved before
        sorted_df['Lagged_Sales'] = np.nan
        sorted_df['Lagged_Available'] = np.nan

        # intialize last row for first iteration. Doesn't really matter what it is
        last = df.loc[0]
        row_len = sorted_df.shape[0]

        for index, row in sorted_df.iterrows():
            lag_date = row["Date_Lagged"]
            # Check if it matches by comparing last weeks value to the compared date
            # And if weekly sales aren't 0
            if((last['Date_Type'] == lag_date) & (last['Weekly_Sales'] > 0)):
                sorted_df.at[index, 'Lagged_Sales'] = last['Weekly_Sales']
                sorted_df.at[index, 'Lagged_Available'] = 1
            else:
                # Fill with median
                sorted_df.at[index, 'Lagged_Sales'] = row['Median Sales']
                sorted_df.at[index, 'Lagged_Available'] = 0

            last = row  # Remember last row for speed
            if(index % int(row_len/10) == 0):  # See progress by printing every 10% interval
                print(str(int(index*100/row_len))+'% loaded')

        return sorted_df

    def get_lagged_variables(self, df):
        # Lagged dates
        df['Date_Lagged'] = df['Date_Type'] - timedelta(days=7)

        # Make a sorted dataframe. This will allow us to find lagged variables much faster!
        sorted_df = df.sort_values(
            ['Store', 'Dept', 'Date_Type'], ascending=[1, 1, 1])
        sorted_df = sorted_df.reset_index(drop=True)

        # Get lagged sales
        sorted_df = self.get_lagged_sales(df, sorted_df)

        # Merge sorted_df to df
        df = df.merge(sorted_df[['Dept', 'Store', 'Date_Type', 'Lagged_Sales',
                                 'Lagged_Available']], how='inner', on=['Dept', 'Store', 'Date_Type'])

        # Calculated sales difference
        df['Sales_diff'] = df['Median Sales'] - df['Lagged_Sales']

        # Variable to be forecasted will be Difference from the median
        df['Difference'] = df['Median Sales'] - df['Weekly_Sales']

        return df

    def remove_unwanted_features(self, df):
        unwanted_cols = ["Store", "Dept", "Date",
                         "Date_Type", "Key", "Weekly_Sales", "Type", "Month", "Date_Lagged"]

        wanted_cols = []

        for col in list(df.columns.values):
            if not col in unwanted_cols:
                wanted_cols.append(col)

        print(wanted_cols)
        final_df = df[wanted_cols]

        return final_df

    def preprocess(self, df):
        df = super().fill_missing_values(df)

        df = self.get_date_features(df)

        df = super().get_encoded_features(df)

        df = self.encode_other_features(df)

        df = self.replace_boolean_with_int(df)

        df = self.get_median_sales(df)

        df = self.get_lagged_variables(df)

        df = self.remove_unwanted_features(df)
        print(df.shape)
        print(df.head())
        return df
