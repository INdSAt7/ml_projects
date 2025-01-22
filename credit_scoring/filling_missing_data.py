import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer


def _filling_missing_float_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing data in the training dataset using KNN for numerical columns
    Returns the modified dataframe.

    :param df: Input dataframe
    :param inplace: If True, modifies the dataframe in place
    :return: DataFrame with filled missing values
    """
    numerical_columns = df.select_dtypes(include=['float32']).columns.tolist()

    if 'target' in numerical_columns:
        numerical_columns.remove('target')
    
    if numerical_columns:
        knn_imputer = KNNImputer(n_neighbors=5, missing_values=-1)
        df[numerical_columns] = pd.DataFrame(
            knn_imputer.fit_transform(df[numerical_columns]),
            columns=numerical_columns
        )


def _filling_type_deposit(df):
    df[df['deposite_type'] == -1]['deposite_type'] = 'дом'

def _filling_direct_deposite(df):
    df['direct_deposite'] = np.where(df['deposite_type'] == 'дом', 1, 0)


def _filling_missing_categorical_data(df: pd.DataFrame):
    _filling_type_deposit(df)
    _filling_direct_deposite(df)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    categorical_columns.remove('deposite_type')

    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='most_frequent', missing_values=-1)
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])


def filling_missing_data(df, inplace=False):
    if not inplace:
        df = df.copy()

    _filling_missing_float_data(df)
    _filling_missing_categorical_data(df)



def filling_missing_data_t(df: pd.DataFrame, inplace=False) -> None:
    """
    Fills missing data in the test dataset similarly to the training dataset.

    :param df: Input dataframe
    :param inplace: If True, modifies the dataframe in place
    """
    if not inplace:
        df = df.copy()

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute missing values for numerical columns using KNN
    if numerical_columns:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numerical_columns] = knn_imputer.fit_transform(df[numerical_columns])

    # Impute missing values for categorical columns with a constant value
    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    # Save the resulting dataframe to a CSV file
    output_path = 'src/test_filled.csv'
    df.to_csv(output_path, index=False)

