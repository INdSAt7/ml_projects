import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

from imblearn.under_sampling import RandomUnderSampler


def undersamle_data(X, y) -> pd.DataFrame:
    rus = RandomUnderSampler(random_state=42)   

    X_resampled, y_resampled = rus.fit_resample(X, y)
    df_resampled = pd.DataFrame(data=X_resampled)
    df_resampled['target'] = y_resampled
    return df_resampled


def read_files(path) -> pd.DataFrame | None:
    filenames = glob.glob(path + "/*.csv")
    data_files = []

    for filename in filenames:
        data_files.append(pd.read_csv(filename))

    data = pd.concat(data_files, ignore_index=True)

    data.drop(['smpl'], axis=1, inplace=True)

    return data


def create_time_series_dict(df):
    time_series = {}

    for row in tqdm(df.to_dict("records")):
        time_series[row['id']] = {
            "time_series": pd.Series(data=row['values'], index=row['features']),
            "target": row['target']
        }

    return time_series


def df_transforming(_df) -> pd.DataFrame:
    df = _df.copy()

    features_array = np.arange(0, len(df.columns)-2)

    df['features'] = [features_array] * len(df)

    df['values'] = tqdm(df.iloc[:, 1:-2].apply(lambda row: np.array(row.values), axis=1))

    if 'target' not in set(df.columns):
        df['target'] = -1

    return df


def df_processing(_df, inplace=False):
    df = _df if inplace else _df.copy()

    category_threshold = 0.05 * len(df)

    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)

        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype('int')

        if df[col].nunique() <= category_threshold:
            df[col] = df[col].astype('category')

    return None if inplace else (df)


def split_dataframe_by_condition(df, condition):
    """
    Разделяет DataFrame на два DataFrame по указанному условию, 
    сохраняя столбцы 'id' и 'target', если они присутствуют.

    Аргументы:
    - df: Исходный DataFrame
    - condition: Условие фильтрации, переданное в виде функции, которая принимает DataFrame и возвращает булев массив

    Возвращает:
    - filtered_df: DataFrame с отфильтрованными столбцами, включая 'id'
    - remaining_df: DataFrame с оставшимися столбцами, также включая 'id'
    """
    # Применяем условие фильтрации к столбцам
    filtered_columns = df.columns[condition(df)]

    print(*filtered_columns)

    # Убедимся, что столбец 'id' включен в filtered_columns, если его нет
    if 'id' not in filtered_columns:
        filtered_columns = ['id'] + list(filtered_columns)

    # Создаем DataFrame с отфильтрованными столбцами
    filtered_df = df[filtered_columns]

    # Определяем оставшиеся столбцы
    remaining_columns = [col for col in df.columns if col not in filtered_columns]

    # Убедимся, что столбец 'id' включен в remaining_columns, если его нет
    if 'id' not in remaining_columns:
        remaining_columns = ['id'] + remaining_columns

    # Создаем DataFrame с оставшимися столбцами
    remaining_df = df[remaining_columns]

    return filtered_df, remaining_df
