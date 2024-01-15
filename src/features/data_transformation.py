from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
from sklearn.impute import SimpleImputer

def data_preprocess(train_df):
    le = LabelEncoder()
    min_max_scl = MinMaxScaler()
    list = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score',
            'Var_1']
    min_max_scaler_columns = ['Age', 'Work_Experience', 'Family_Size']

    for i in list:
        train_df[i] = le.fit_transform(train_df[i])

    for i in min_max_scaler_columns:
        train_df[i] = min_max_scl.fit_transform(train_df[i].values.reshape(-1, 1))
    return train_df

class DataTransformation:
    def initiate_data_transformation(self, train_df, test_df):
        train_df = data_preprocess(train_df)
        X = train_df.drop(columns=["Segmentation"], axis=1)
        y = train_df["Segmentation"]
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        return (
            X_train, y_train,
            X_test, y_test
        )
