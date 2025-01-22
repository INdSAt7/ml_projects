from typing import Optional, Dict, Tuple, Union
# import numbers
# import portion
import pandas as pd
import numpy as np

# interval_dict_uint = portion.IntervalDict({
#     portion.closed(0, 2 ** 8 - 1): 'np.uint8',
#     portion.closed(0, 2 ** 16 - 1): 'np.uint16',
#     portion.closed(0, 2 ** 32 - 1): 'np.uint32',
#     portion.closed(0, 2 ** 64 - 1): 'np.uint64'
# })

# interval_dict_int = portion.IntervalDict({
#     portion.closed(-2 ** 7, 2 ** 7 - 1): 'np.int8',
#     portion.closed(-2 ** 15, 2 ** 15 - 1): 'np.int16',
#     portion.closed(-2 ** 31, 2 ** 31 - 1): 'np.int32',
#     portion.closed(-2 ** 63, 2 ** 63 - 1): 'np.int64'
# })

BYTES_IN_MEGABYTE = 1024 ** 2


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
        usage_mb = usage_b / BYTES_IN_MEGABYTE
        return usage_mb
    elif isinstance(pandas_obj, pd.Series):
        usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / BYTES_IN_MEGABYTE
        return usage_mb
    else:
        raise TypeError(f"Input must be a DataFrame or Series object, not {type(pandas_obj)}")


def _optimize_integer_cols(df: pd.DataFrame, col: str) -> None:
    """Downcasts integer columns to the smallest possible subtype."""
    df[col] = df[col].fillna(int(-1))

    # min_val, max_val = df[col].min(), df[col].max()

    # if min_val >= 0:
    #     dtype = interval_dict_uint.get(max_val, None)
    # else:
    #     dtype = interval_dict_int.get(portion.closed(min_val, max_val), None)

    # if dtype:
    df[col] = df[col].astype(np.int8)
    # else:
    #     df[col] = pd.to_numeric(df[col], downcast='integer')


def _optimize_float_cols(df: pd.DataFrame, col: str, target_dtype='float32') -> None:
    """Downcasts float columns and converts whole numbers to integers if possible."""
    df[col] = df[col].fillna(float(-1))

    if (df[col] % 1 == 0).all():  # Если все значения целые
        _optimize_integer_cols(df, col)
    else:
        df[col] = df[col].astype(target_dtype)


def _optimize_categorical_cols(df: pd.DataFrame, col: str) -> None:
    """Encodes categorical columns with limited unique values to integers or bool for binary categories."""
    df[col] = df[col].fillna(-1)
    df[col] = df[col].astype('category')


def auto_optimize_dtypes(
        df_: pd.DataFrame,
        inplace=False,
        ratio_unique=0.0037,
        float_dtype: str = 'float32',
        custom_dtype: Optional[Dict[str, str]] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Dict]]:
    """Automatically optimizes numerical, categorical, and boolean columns for minimal memory usage.
       Downcasts numerical types, encodes categorical strings, and converts binary categories to bool.

       :param df_: DataFrame to optimize
       :param inplace: if False, returns a new optimized DataFrame; if True, modifies in place
       :param ratio_unique: threshold ratio of unique values in a column to convert to categorical
       :param float_dtype: target float type, e.g., 'float32' or 'float16'
       :param custom_dtype: dictionary with custom dtype assignments for specific columns

       :return: Tuple containing the optimized DataFrame (or None if inplace=True)
                and a dictionary of mappings for each transformed categorical column.
    """
    if not (0 < ratio_unique <= 1):
        raise ValueError(f"ratio_unique should be between 0 and 1, not {ratio_unique}")

    df = df_ if inplace else df_.copy()

    for col in df.columns:
        # if custom_dtype and col in custom_dtype:
        #     df[col] = df[col].astype(custom_dtype[col])
        #     continue

        # if (df[col].nunique() == 2) and (df[col].dtype != 'category'):  # Convert to bool if only two unique values
        #     df[col] = df[col].astype(bool)
        #     continue

        if pd.api.types.is_float_dtype(df[col]):
            _optimize_float_cols(df, col, float_dtype)

        elif pd.api.types.is_integer_dtype(df[col]):
            _optimize_integer_cols(df, col)

        if df[col].nunique() < len(df) * ratio_unique:  #df[col].dtype == 'object' and
            _optimize_categorical_cols(df, col)


    return None if inplace else df
