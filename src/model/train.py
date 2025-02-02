import argparse
import glob
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main(args):
    # Enable autologging
    
    mlflow.autolog()
    # Read data
    df = get_csvs_df(args.training_data)
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

def get_csvs_df(path):

    
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    """Splits the dataset into training and test sets."""
    
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness',
            'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=0)

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    """Trains a logistic regression model and logs parameters."""
    
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model

def parse_args():

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    return parser.parse_args()

if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")
