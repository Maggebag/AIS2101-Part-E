import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def import_and_split_data(csv_file, test_size=0.2, random_state=42):
    # Load the dataset from CSV file
    dataset = pd.read_csv(csv_file)

    # Split features (X) and target (y)
    X = dataset.drop('Class', axis=1)  # Features
    y = dataset['Class']  # Target

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def import_data_no_class(csv_file):
    # Load the dataset from CSV file
    dataset = pd.read_csv(csv_file)

    # Split features (X) and target (y)
    X = dataset.drop('Class', axis=1)  # Features
    y = dataset['Class']  # Target

    return X, y, dataset

def import_data_normalized_split(csv_file, test_size=0.2, random_state=42):
    # Load the dataset from CSV file
    dataset = pd.read_csv(csv_file)

    # Split features (X) and target (y)
    X = dataset.drop('Class', axis=1)  # Features
    y = dataset['Class']  # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, dataset