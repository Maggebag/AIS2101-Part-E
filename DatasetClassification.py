import pandas as pd
from scipy.io import arff
from plotting import *


def remove_outliers(dataset_to_search):
    all_outliers = pd.DataFrame()
    for col in dataset_to_search.columns:
        if dataset_to_search[col].dtype != 'object':  # Exclude non-numeric columns
            quart1 = dataset_to_search[col].quantile(0.25)
            quart3 = dataset_to_search[col].quantile(0.75)
            IQR = quart3 - quart1  # Inter-quartile range
            low_val = quart1 - 1.5 * IQR
            high_val = quart3 + 1.5 * IQR
            col_outliers = dataset_to_search.loc[(dataset_to_search[col] < low_val) | (dataset_to_search[col] > high_val)]
            all_outliers = pd.concat([all_outliers, col_outliers])
    dataset_cleaned = dataset_to_search.loc[~dataset_to_search.index.isin(all_outliers.index)]
    print("Total number of outliers found and removed:", all_outliers.shape[0])
    return dataset_cleaned


data = arff.loadarff('dataset/Rice_Cammeo_Osmancik.arff')
dataset = pd.DataFrame(data[0])  # Dataset is a list of dictionaries
#dataset.info()

# Test for duplicated values
if dataset.duplicated().any():
    dataset = dataset.drop_duplicates()
else:
    print("No duplicated rows found")

# Dataset should not have any missing values, but we test anyway
if dataset.isnull().any().any():
    dataset = dataset.dropna()
else:
    print("No missing values found")

#print(dataset.describe())

new_dataset = remove_outliers(dataset)
#print(outliers_found)

#print(new_dataset.describe())

plot_all(new_dataset)


