import numpy as np
import pandas as pd
import mass_ts as mts
import timeit
from IPython.display import display

from modules.distance_profile import brute_force
from modules.bestmatch import NaiveBestMatchFinder, UCR_DTW
from modules.plots import mplot2d


def _get_param_values(exp_params: dict, param: str) -> list:
    """
    Get experiment parameter values
    
    Parameters
    ----------
    exp_params: experiment parameters
    param: parameter name
    
    Returns
    -------
    param_values: experiment parameter values
    """

    if (param in exp_params['fixed'].keys()):
        param_values = [exp_params['fixed'][param]]
    else:
        param_values = exp_params['varying'][param]

    return param_values


def _run_experiment_dist_profile(algorithm: str, data: dict, exp_params: dict, alg_params: dict) -> np.ndarray:
    """
    Run an experiment to measure the execution time of an algorithm which calculates distance profile between time series and query
    
    Parameters
    ----------
    algorithm: algorithm name
    data: set of time series and queries
    exp_params: experiment parameters
    alg_params: algorithm parameters
    
    Returns
    -------
    times: execution times of algorithm
    """
    
    n_list = _get_param_values(exp_params, 'n')
    m_list = _get_param_values(exp_params, 'm')

    times = []

    for n in n_list:
        for m in m_list:
            match algorithm:
                case 'brute_force':
                    runtime_code = f"brute_force(data['ts']['{n}'], data['query']['{m}'])"
                case 'mass3': 
                    runtime_code = f"mts.mass3(data['ts']['{n}'], data['query']['{m}'], alg_params['segment_len'])"
                case 'mass' | 'mass2':
                    runtime_code = f"mts.{algorithm}(data['ts']['{n}'], data['query']['{m}'])"    
            try:
                time = timeit.timeit(stmt=runtime_code, number=1, globals={**globals(), **locals()})
            except:
                time = np.nan

            times.append(time)
    
    return np.array(times)


def _run_experiment_best_match(algorithm: str, data: dict, exp_params: dict, alg_params: dict) -> np.ndarray:
    """
    Run an experiment to measure the execution time of an best match algorithm
    
    Parameters
    ----------
    algorithm: algorithm name
    data: set of time series and queries
    exp_params: experiment parameters
    alg_params: algorithm parameters
    
    Returns
    -------
    times: execution times of algorithm
    """
    
    n_list = _get_param_values(exp_params, 'n')
    m_list = _get_param_values(exp_params, 'm')
    r_list = _get_param_values(exp_params, 'r')

    times = []

    for r in r_list:
        r_times = []
        for n in n_list:
            for m in m_list:
                match algorithm:
                    case 'naive':
                        naive_bestmatch_model = NaiveBestMatchFinder(alg_params['excl_zone_frac'], alg_params['topK'], alg_params['normalize'], r)
                        runtime_code = f"naive_bestmatch_model.perform(data['ts']['{n}'], data['query']['{m}'])"
                    case 'ucr-dtw':
                        ucr_dtw_bestmatch_model = UCR_DTW(alg_params['excl_zone_frac'], alg_params['topK'], alg_params['normalize'], r)
                        runtime_code = f"ucr_dtw_bestmatch_model.perform(data['ts']['{n}'], data['query']['{m}'])"

                try:
                    time = timeit.timeit(stmt=runtime_code, number=1, globals={**globals(), **locals()})
                except:
                    time = np.nan

                r_times.append(time)

        times.append(r_times)

    return np.array(times)


def run_experiment(algorithm: str, task: str, data: dict, exp_params: dict, alg_params: dict = None) -> np.ndarray:
    """
    Run an experiment to measure the execution time of an algorithm
    
    Parameters
    ----------
    algorithm: algorithm name
    task: the task that the algorithm performs
    data: set of time series and queries
    exp_params: experiment parameters
    alg_params: algorithm parameters
    
    Returns
    -------
    times: execution times of algorithm
    """
    
    if (task == "distance_profile"):
        times = _run_experiment_dist_profile(algorithm, data, exp_params, alg_params)
    elif (task == "best_match"):
        times = _run_experiment_best_match(algorithm, data, exp_params, alg_params)
    else:
        raise NotImplementedError

    return times


def visualize_plot_times(times: np.ndarray, comparison_param: np.ndarray, exp_params: dict) -> None:
    """
    Visualize plot with execution times
    
    Parameters
    ----------
    times: execution times of algorithms
    comparison_param: name of comparison parameter
    exp_params: experiment parameters
    """

    if ('n' in exp_params['varying'].keys()):
        varying_param_name = 'Time series length'   
        varying_param_value = exp_params['varying']['n']
    else:
        varying_param_name = 'Query length'
        varying_param_value = exp_params['varying']['m']

    plot_title = 'Runtime depending on ' + varying_param_name 
    trace_titles = comparison_param
    x_axis_title = varying_param_name
    y_axis_title = 'Runtime, s'
    
    mplot2d(np.array(varying_param_value), times, plot_title, x_axis_title, y_axis_title, trace_titles)


def calculate_speedup(base_algorithm_times: np.ndarray, improved_algorithms_times: np.ndarray) -> np.ndarray:
    """
    Calculate speedup algorithms relative to the base algorithm by formula: speedup = base_algorithm_times/algorithms_times
    
    Parameters
    ----------
    base_algorithm_times: execution times of the base algorithm
    algorithms_times: execution times of algorithms for which speedup is calculated
    
    Returns
    -------
    speedup: speedup algorithms relative to the base algorithm
    """

    speedup = base_algorithm_times/improved_algorithms_times

    return speedup


def visualize_table_speedup(speedup_data: np.ndarray, table_index: list, table_columns: list, table_caption: str) -> None:
    """
    Visualize the table with speedup
    
    Parameters
    ----------
    data: input data of table
    table_index: index of table
    table_columns: names of table columns
    table_title: title of table
    """

    df = pd.DataFrame(data=speedup_data, index=table_index, columns=table_columns)

    def style_negative(value, props=''):
        return props if value < 1 else None

    def style_positive(value, props=''):
        return props if value >= 1 else None

    style_df = df.style.map(style_negative, props='color: red;')\
                .map(style_positive, props='color: green;')\
                .set_properties(**{'border': '1px black solid !important', 'text-align': 'center'})\
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('border','1px black solid !important'), ('text-align', 'center')]
                    },
                    {'selector': 'caption',
                    'props': [('font-size', '16px'),
                                ('font-weight', 'bold'),
                                ('padding', '10px 0px 10px 0px')
                            ]
                    }
                ])\
                .set_caption(table_caption)

    display(style_df)
