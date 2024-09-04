import numpy as np


def is_nan_inf(arr: np.ndarray) -> bool:
    """
    Check if the array contains np.nan, -np.nan, or np.inf values

    Parameters
    ----------
    arr: array

    Returns
    -------
        flag of checking if the array contains np.nan, -np.nan, or np.inf values
    """

    return np.isnan(arr) or np.isinf(abs(arr))


def apply_exclusion_zone(a: np.ndarray, idx: int, excl_zone: int, val: float) -> np.ndarray:
    """ 
    Set all values of array to `val` in a window around a given index

    Parameters
    ----------
    a: array
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    val: the elements within the exclusion zone will be set to this value

    Returns
    -------
    a: array that is applied an exclusion zone
    """
    
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)

    a[zone_start : zone_stop + 1] = val

    return a
