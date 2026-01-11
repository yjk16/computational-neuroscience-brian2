"""
Docstring for allen_functions.py

Functions used related to the data from the Allen Institute
"""
import numpy as np
import pandas as pd
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

# this saves the NWB file to 'cell_types/specimen_464212183/ephys.nwb'
cell_specimen_id =    525011903
data_set = ctc.get_ephys_data(cell_specimen_id)


def get_sweep_data(sweep_no):
    """
    Retrieve the data for a specified sweep number.
    
    Parameters
    ----------
    sweep_no : int
        The specified sweep number.

    Returns
    -------
    sweep_data : dict
        A dictionary containing arrays for the keys:
        'stimulus', 'response', 'stimulus_unit', 'index_range', and 'sampling_rate'.
    """
    sweep_data = data_set.get_sweep(sweep_no)
    return sweep_data


def get_index_range_from_sweep(sweep_number):
    """
    Retrieve the index range for a specified sweep number.
    
    Parameters
    ----------
    sweep_number : int
        The specified sweep number.

    Returns
    -------
    index_range : tuple
        Tuple containing the start and end indices of sweep.
    """

    sweep_data = data_set.get_sweep(sweep_number)
    index_range = sweep_data["index_range"]
    return index_range


def get_sampling_rate_from_sweep(sweep_number):
    """
    Retrieve the sampling rate for a specified sweep number.
    
    Parameters
    ----------
    sweep_number : int
        The specified sweep number.

    Returns
    -------
    sampling_rate : float
        The sampling rate of the sweep (in Hz).
    """
    sweep_data = data_set.get_sweep(sweep_number)
    sampling_rate = sweep_data["sampling_rate"] # in Hz
    return sampling_rate


def get_stimulus_from_sweep(sweep_number):
    """
    Retrieve the stimulus data for a specified sweep number.
    
    Parameters
    ----------
    sweep_number : int
        The specified sweep number.

    Returns
    -------
    input_current : numpy.ndarray
        The stimulus data of the sweep (in pA).
    """
    sweep_data = data_set.get_sweep(sweep_number)
    index_range = sweep_data["index_range"]
    input_current = sweep_data["stimulus"][0:index_range[1]+1] * 10**12  # in pA

    return input_current


def get_response_from_sweep(sweep_number):
    """
    Retrieve the response data for a specified sweep number.
    
    Parameters
    ----------
    sweep_number : int
        The specified sweep number.

    Returns
    -------
    output_response : numpy.ndarray
        The response data of the sweep (in mV).
    """
    sweep_data = data_set.get_sweep(sweep_number) 
    index_range = sweep_data["index_range"]
    output_response = sweep_data["response"][0:index_range[1]+1] * 1000 # in mV

    return output_response


def get_stim_res_from_sweep(sweep_number):
    """
    Retrieve the input stimulus and output response data for a specified sweep number.
    
    Parameters
    ----------
    sweep_number : int
        The specified sweep number.

    Returns
    -------
        input_current, output_response : tuple of numpy.ndarray
            Tuple containing:
            - input_current : numpy.ndarray
                Input stimulus (pA)
            - output_response: numpy.ndarray
                Output response (mV).
    """
    input_current = get_stimulus_from_sweep(sweep_number)
    output_response = get_response_from_sweep(sweep_number)
    return input_current, output_response


def make_array_of_data_for_sweeps(start, end, slicer):
    """
    Create a sliced dataset for input stimulus and output response.
    
    Parameters
    ----------
    start : int
        The start index.
    end : int
        The end index.
    slicer : int
        The numeric value for slicing

    Returns
    -------
    Tuple containing:
    - stimulus_list : list of numpy.ndarray
        Input stimulus for each sweep (pA).
    - response_list: list of numpy.ndarray
        Output response for each sweep (mV).
    """
    stimulus_list = []
    response_list = []

    for sweep in range(start,end,slicer):

        this_stimulus = get_stimulus_from_sweep(sweep)
        this_response = get_response_from_sweep(sweep)

        stimulus_list.append(this_stimulus)
        response_list.append(this_response)

    return stimulus_list, response_list


def turn_lists_to_csvs(datain, dataout, filein, fileout):
    """
    Convert lists of input stimulus and output response to csvs.
    
    Parameters
    ----------
    datain : list
        List of input stimulus arrays.
    dataout : list
        List of output response arrays.
    filein : str
        File path for saving input stimulus CSV.
    fileout : str
        File path for saving output response CSV.
    """
    in_np = np.array(datain)
    out_np = np.array(dataout)

    in_pd = pd.DataFrame(in_np)
    out_pd = pd.DataFrame(out_np)

    in_pd.to_csv(filein)
    out_pd.to_csv(fileout)
