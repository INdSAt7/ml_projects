import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import functools
from typing import Dict, List, Tuple, Union, Callable

from statsmodels.tsa.stattools import acf, pacf
import ruptures as rpt

def get_ma_features(window_sizes: List[int], time_series: List[float]) -> Dict[str, float]:
  """
  Calculates moving average features for a given time series.

  Args:
      window_sizes: A list of window sizes to calculate moving averages for.
      time_series: The time series data.

  Returns:
      A dictionary containing the calculated moving averages, keyed by the window size.
  """
  moving_averages = {}
  for window_size in window_sizes:
      window = time_series[ (len(time_series) - window_size):]
      ma = sum(window) / len(window)
      moving_averages[f'MA{window_size}'] = ma

  return moving_averages

def get_fn_features(dict_of_fns: Dict[str, callable], time_series: List[float]) -> Dict[str, float]:
    """
    Applies a dictionary of functions to a time series and returns the results as a dictionary.

    Args:
        dict_of_fns: A dictionary of functions to apply to the time series. The keys are the feature names and the values are the functions.
        time_series: The time series data.

    Returns:
        A dictionary containing the results of applying the functions to the time series.
    """

    features = {}

    for fn_name, fn in dict_of_fns.items():
        features[fn_name] = fn(time_series)

    return features

def get_fourier_features(n: int, time_series: List[float]) -> Dict[str, float]:
    """
    Calculates Fourier transform features for a given time series.

    Args:
        n: The number of Fourier coefficients to extract.
        time_series: The time series data.

    Returns:
        A dictionary containing the first n Fourier coefficients, keyed by their index.
    """

    N = len(time_series)
    T = 1.0   
    yf = np.fft.fft(time_series)
    xf = np.fft.fftfreq(N, T)[:N//2]   
    yf_normalized = 2.0/N * np.abs(yf[:N//2])

    fourier_dict = {f"F{i}": f for i, f in enumerate(yf_normalized[:n])}

    return fourier_dict

def get_changepoint_features(penalty: float, time_series: List[float]) -> Dict[str, int]:
    """
    Detects change points in a time series using the PELT algorithm and returns the number of change points.

    Args:
        penalty: The penalty parameter for the PELT algorithm.
        time_series: The time series data.

    Returns:
        A dictionary containing the number of change points detected in the time series.
    """

    time_series = np.array(time_series)

    algo = rpt.Pelt(model='l2').fit(time_series)

    change_points = algo.predict(pen=penalty)

    num_change_points = len(change_points) - 1  

    return {'number_of_cp': num_change_points} #, change_points[:-1]

def get_acf_pacf_features(lag: int, time_series: List[float]) -> Dict[str, float]:
    """
    Calculates autocorrelation (ACF) and partial autocorrelation (PACF) features for a given time series.

    Args:
        lag: The maximum lag to calculate ACF and PACF for.
        time_series: The time series data.

    Returns:
        A dictionary containing the ACF and PACF values for lags 1 to lag, keyed by their respective names.
    """

    results = {}

    acf_values = acf(time_series, nlags=lag)
    for i in range(1, lag + 1):
        results[f'ACF{i}'] = acf_values[i]

    pacf_values = pacf(time_series, nlags=lag)
    for i in range(1, lag + 1):
        results[f'PACF{i}'] = pacf_values[i]

    return results

def get_acf_pacf_features(lag: int, time_series: List[float]) -> Dict[str, float]:
    """
    Calculates autocorrelation (ACF) and partial autocorrelation (PACF) features for a given time series.

    Args:
        lag: The maximum lag to calculate ACF and PACF for.
        time_series: The time series data.

    Returns:
        A dictionary containing the ACF and PACF values for lags 1 to lag, keyed by their respective names.
    """

    results = {}

    acf_values = acf(time_series, nlags=lag)
    for i in range(1, lag + 1):
        results[f'ACF{i}'] = acf_values[i]

    pacf_values = pacf(time_series, nlags=lag)
    for i in range(1, lag + 1):
        results[f'PACF{i}'] = pacf_values[i]

    return results

def get_lr_features(time_series: List[float]) -> Dict[str, float]:
    """
    Calculates the coefficients of a linear regression model fitted to the time series.

    Args:
        time_series: The time series data.

    Returns:
        A dictionary containing the intercept (beta0) and slope (beta1) of the linear regression model.
    """

    t = np.arange(1, len(time_series) + 1)

    beta1 = np.sum((t - np.mean(t)) * (time_series - np.mean(time_series))) / np.sum((t - np.mean(t)) ** 2)
    beta0 = np.mean(time_series) - beta1 * np.mean(t)

    return {"beta0": beta0, "beta1": beta1}

def get_outlier_features(k: float, time_series: List[float]) -> Dict[str, int]:
    """
    Calculates the number of outliers based on Tukey's method.

    Args:
        k: The multiplier for the interquartile range (IQR).
        time_series: The time series data.

    Returns:
        A dictionary containing the number of low and high outliers.
    """

    Q1 = np.percentile(time_series, 25)
    Q3 = np.percentile(time_series, 75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    low_outliers = time_series[time_series < lower_bound]
    high_outliers = time_series[time_series > upper_bound]
    
    outliers_dict = {
        "tukey_low_outliers": len(low_outliers),
        "tukey_high_outliers": len(high_outliers)
    }
    
    return outliers_dict


def get_quantiles(time_series: List[float]) -> Dict[str, int]:
    """
    Calculates 25 and 75 percentile.

    Args:
        k: The multiplier for the interquartile range (IQR).
        time_series: The time series data.

    Returns:
        A dictionary containing 25 and 75 percentile.
    """

    Q1 = np.percentile(time_series, 25)
    Q3 = np.percentile(time_series, 75)

    quantiles_dict = {
        "Q25": Q1,
        "Q75": Q3
    }

    return quantiles_dict


def get_all_features(time_series: List[float], list_of_feature_fns: List) -> Dict[str, Union[float, int]]:
    """
    Calculates all features for a given time series using a list of feature functions.

    Args:
        time_series: The time series data.
        list_of_feature_fns: A list of feature functions to apply to the time series.

    Returns:
        A dictionary containing all calculated features.
    """

    features = {}

    for fn in list_of_feature_fns:
        features.update(fn(time_series))

    return features

def construct_features_df(list_of_ts: Dict[int, Dict[str, List[float]]], list_of_feature_fns: List[callable]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs a DataFrame of features from a list of time series.

    Args:
        list_of_ts: A dictionary of time series data, where the keys are the time series IDs and the values are dictionaries containing the time series data and the target.
        list_of_feature_fns: A list of functions that extract features from a time series.

    Returns:
        A tuple containing a DataFrame of features and a Series of targets.
    """
    features = []
    y = []

    for ts_id, ts_data in tqdm(list_of_ts.items()):
        features.append(get_all_features(ts_data['time_series'], list_of_feature_fns))
        y.append(ts_data['target'])

    return pd.DataFrame(features), pd.Series(y)

def process_ts(list_of_feature_fns: List[Callable], ts_data: Tuple[int, Dict[str, pd.Series]]) -> Tuple[int, Dict[str, float], int]:
    """
    Processes a single time series to extract features and target.

    Args:
        list_of_feature_fns: A list of functions that extract features from a time series.
        ts_data: A tuple containing the time series ID and the time series data.

    Returns:
        A tuple containing the time series ID, extracted features, and the target.
    """
    ts_id, data = ts_data
    features = get_all_features(data['time_series'], list_of_feature_fns)
    return ts_id, features, data['target']

def construct_features_df_parallel(list_of_ts: Dict[int, Dict[str, pd.Series]],
                                   list_of_feature_fns: List[Callable]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs a DataFrame of features from a dictionary of time series in parallel.

    Args:
        list_of_ts: A dictionary of time series data, where the keys are the time series IDs, 
                    and the values are dictionaries containing the time series data and the target.
        list_of_feature_fns: A list of functions that extract features from a time series.

    Returns:
        A tuple containing a DataFrame of features (with time series IDs as index) and a Series of targets.
    """
    process_ts_features = functools.partial(process_ts, list_of_feature_fns)

    with Pool(processes=cpu_count() - 2) as pool:
        results = list(tqdm(pool.imap(process_ts_features, list_of_ts.items()), total=len(list_of_ts)))

    # Separate out the data for DataFrame and Series
    indices = [result[0] for result in results]
    features = [result[1] for result in results]
    targets = [result[2] for result in results]

    # Construct DataFrame and Series with indices
    features_df = pd.DataFrame(features, index=indices)
    targets_series = pd.Series(targets, index=indices)

    return features_df, targets_series
