import os
import pickle
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from utility import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join( "../artifacts", "model.pkl")
    print("trained_model_file_path", trained_model_file_path)


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        lr = LogisticRegression()
        mnb = MultinomialNB()
        knn = KNeighborsClassifier()
        dtc = DecisionTreeClassifier(criterion='gini')
        dtcr = DecisionTreeClassifier(criterion='entropy')

        models = {'LogisticRegression': lr, 'MultinomialNB': mnb, 'KNeighborsClassifier': knn, 'DecisionTreeClassifier(gini)': dtc,
                  'DecisionTreeClassifier(entropy)': dtcr}

        best_model_name = None
        best_accuracy = 0.0

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            mpred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, mpred)

            print(f"Model: {model_name}")
            print(f"Training Score: {train_score}")
            print(f"Testing Accuracy: {test_accuracy}")
            print("Classification Report:\n", classification_report(y_test, mpred))
            print("\n")

            # Track the best model based on testing accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_name = model_name

        print(f"The best model is: {best_model_name} with accuracy: {best_accuracy}")
        best_model = models[best_model_name]
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)
        return r2_square


def run_train_pipeline():
    obj = DataIngestion()
    train_df, test_df = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_df, test_df)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    run_train_pipeline()