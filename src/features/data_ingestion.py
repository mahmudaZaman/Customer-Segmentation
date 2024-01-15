import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer


@dataclass
class DataIngestionConfig:
    train_data_uri: str = "/Users/shuchi/Documents/work/personal/Customer-Segmentation/dataset/train.csv"
    print("train_data_uri", train_data_uri)
    test_data_uri: str = "/Users/shuchi/Documents/work/personal/Customer-Segmentation/dataset/test.csv"
    print("test_data_uri", test_data_uri)

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        train_df = pd.read_csv(self.ingestion_config.train_data_uri, index_col='ID')
        test_df = pd.read_csv(self.ingestion_config.test_data_uri)

        imp = SimpleImputer(strategy='most_frequent')
        list_cat = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']
        list_num = ['Work_Experience', 'Family_Size']

        for i in list_cat:
            train_df[i] = imp.fit_transform(train_df[i].values.reshape(-1, 1))
        for i in list_num:
            train_df[i] = imp.fit_transform(train_df[i].values.reshape(-1, 1))

        return (
            train_df, test_df
        )
