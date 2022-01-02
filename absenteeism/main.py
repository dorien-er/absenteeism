import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

# create class to predict absenteeism from new data
class absenteeism_model():

    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files which were saved
        self.reg = model_file
        self.scaler = scaler_file
        self.data = None

    # take a data file (*.csv) and preprocess it
    def load_and_clean_data(self, data_file):

        # import the data
        df = pd.read_csv(data_file,delimiter=',')
        # drop the 'ID'column
        df = df.drop(['ID'], axis = 1)

        # create a separate dataframe, containing dummy values for ALL avaiable reasons
        ohe = OneHotEncoder(sparse = False) # Instanciate encoder
        ohe.fit(df[['Reason for Absence']]) # Fit encoder
        absence_encoded = ohe.transform(df[['Reason for Absence']]) # Encode reason for absence
        absence_df = pd.DataFrame(absence_encoded, dtype = 'int')
        absence_df.drop(columns =0, inplace = True)

        # split reason_columns into 4 types
        df["reason 1"] = absence_df.loc[:,1:14].sum(axis=1)
        df["reason 2"] = absence_df.loc[:,15:17].sum(axis=1)
        df["reason 3"] = absence_df.loc[:,18:21].sum(axis=1)
        df["reason 4"] = absence_df.loc[:,22:].sum(axis=1)

        # to avoid multicollinearity, drop the 'Reason for Absence' column from df
        df = df.drop(['Reason for Absence'], axis = 1)

        # convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # create new features called called 'Month' and 'Day of the Week'
        df['Month'] = df['Date'].apply(lambda x: x.month)
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # drop the 'Date' column from df
        df = df.drop(columns = 'Date')

        # map 'Education' variables; the result is a dummy
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # replace the NaN values
        df = df.fillna(value=0)

        # drop the variables we decide we don't need
        df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)

        # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df[['reason 1', 'reason 2', 'reason 3', 'reason 4', 'Education',
       'Transportation Expense', 'Age', 'Body Mass Index', 'Children', 'Pets',
       'Month']].copy()

        # StandardScale numerical features
        X_unscaled = df[['Transportation Expense', 'Age', 'Body Mass Index', 'Children', 'Pets', 'Month']]
        X_scaled = self.scaler.transform(X_unscaled)
        df = df.drop(columns = ['Transportation Expense', 'Age', 'Body Mass Index', 'Children', 'Pets', 'Month'])
        df[['Transportation Expense', 'Age','Body Mass Index', 'Children','Pets', 'Month']] = X_scaled

        # reorder columns
        df = df[['reason 1', 'reason 2', 'reason 3', 'reason 4', 'Education',
       'Transportation Expense', 'Age', 'Body Mass Index', 'Children', 'Pets',
       'Month']]


        # we need this line so we can use it in the next functions
        self.data = df

        return df

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
