from typing import List, Tuple, Dict
import math
from scipy.stats import norm
import numpy as np


def calculate_confidence_interval(metric_values: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * (std / np.sqrt(len(metric_values)))

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def max_diff(time_series: List[float]) -> float:
    if len(time_series) < 2:
        return 0

    max_diff = 0
    for i in range(1, len(time_series)):
        diff = abs(time_series[i] - time_series[i - 1])
        if diff > max_diff:
            max_diff = diff

    return max_diff


def absolute_gain_basis(time_series: List[float]) -> float:
    print(time_series[len(time_series) - 1], time_series[0])
    if len(time_series) < 2:
        return 0.0
    return (time_series[len(time_series) - 1] - time_series[0])


def growth_coef_basis(time_series: List[float]) -> float:
    if len(time_series) < 2 or time_series[0] == 0:
        return float('nan')
    return (time_series[len(time_series) - 1] / time_series[0]) - 1


def average_gain_rate(time_series: List[float]) -> float:
    if len(time_series) < 2:
        return 0.0
    gains = [time_series[i] - time_series[i - 1] for i in range(1, len(time_series))]

    gain_rates = [gains[i] for i in range(1, len(gains)) if time_series[i] != 0]
    return sum(gain_rates) / len(gain_rates) if gain_rates else 0.0


def average_growth_rate(time_series: List[float]) -> float:
    if len(time_series) < 2:
        return 0.0
    growth_rates = []
    for i in range(1, len(time_series)):
        if time_series[i - 1] != 0:
            growth_rates.append((time_series[i] / time_series[i - 1]) - 1)
    return sum(growth_rates) / len(growth_rates) if growth_rates else 0.0


def average_absolute_gain(time_series: List[float]) -> float:
    if len(time_series) < 2:
        return 0.0
    absolute_gains = [abs(time_series[i] - time_series[i - 1]) for i in range(1, len(time_series))]
    return sum(absolute_gains) / len(absolute_gains) if absolute_gains else 0.0
