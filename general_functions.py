"""
Docstring for general_functions.py.

General functions used for converting data, running models and creating plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import *
from brian2modelfitting import *

#####################################################################################################################

def csv_to_np(file):
    """
    Convert a CSV file to a NumPy array

    Parameters
    ----------
    file : str
        The path to the CSV file.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the contents of the CSV file.
    """
    converted_data = pd.read_csv(file, index_col=0).to_numpy()

    return converted_data


def one_array(data, index):
    """
    Return a single array from NumPy data.

    Parameters
    ----------
    data : numpy.ndarray
        Input array.
    index : int
        Index of array along axis 0.

    Returns
    -------
    numpy.ndarray
        Single array corresponding to specified index of input array.
    """
    array = data[index]

    return array


def convert_pA_to_amps(patoamp):
    """
    Convert values from picoamperes (pA) to amperes (A).

    Parameters
    ----------
    patoamp : numpy.ndarray
        Array of values in picoamperes.

    Returns
    -------
    numpy.ndarray
        Array of converted values in amperes.
    """
    converted_pa_to_amps = patoamp * 10 ** -12

    return converted_pa_to_amps


def time_steps(data):
    """
    Find time steps.

    Parameters
    ----------
    data : array_like
        Input data.

    Returns
    -------
    numpy.ndarray
        Time steps.
    """
    time_step = np.arange(0, len(data))
    result = time_step * (1/50)

    return result


def slice_data_start_end(datain, dataout, st=100_000, et=150_000):
    """
    Return sliced input and output data.

    Parameters
    ----------
    datain : array_like
        Array of input values.
    dataout : array_like
        Array of output values.
    st : int
        Start index of the slice.
    et : int
        End index of the slice.

    Returns
    -------
    new_in : numpy.ndarray
        Sliced input data.

    new_out : numpy.ndarray
        Sliced output data.

    new_t_steps : numpy.ndarray
        Array of new time-steps.
    """
    start = st
    end = et

    new_in = datain[start:end]
    new_out = dataout[start:end]
    new_t_steps = time_steps(new_in)

    return new_in, new_out, new_t_steps


def reshape_data(indata, outdata):
    """
    Return reshaped input and output data.

    Parameters
    ----------
    indata : array_like
        Array of input values.
    outdata : array_like
        Array of output values.

    Returns
    -------
    indata_reshaped : numpy.ndarray
        Reshaped input data.

    outdata_reshaped : numpy.ndarray
        Reshaped output data.
    """
    indata_reshaped = np.reshape(indata, (1, -1))
    outdata_reshaped = np.reshape(outdata, (1, -1))

    return indata_reshaped, outdata_reshaped


#####################################################################################################################
# MODELS
#####################################################################################################################

# Code for models sourced from: https://brian2modelfitting.readthedocs.io/en/stable/introduction/tutorial_hh.html


def run_hh(input_current, output_response, cm_var=120, el=-70, ek=-78, ena=40, vt=-50, round_num=10,
           glmin=0, glmax=10, gnamin=1, gnamax=20, gkdmin=0.01, gkdmax=4):
    """
    Run the Hodgkin-Huxley model.

    Parameters
    ----------
    input_current : array_like
        Array of input values.
    output_response : array_like
        Array of output values.
    cm_var : float
        Numeric value for capacitance (pF)
    el : float
        Numeric value for leak reversal potential (mV)
    ek : float
        Numeric value for potassium reversal potential (mV)
    ena : float
        Numeric value for sodium reversal potential (mV)
    vt : float
        Numeric value for spike threshold (mV)
    round_num : int
        Number of rounds
    glmin : float
        Minimum value for leak conductance (nS)
    glmax : float
        Maximum value for leak conductance (nS)
    gnamin : float
        Minimum value for sodium conductance (uS)
    gnamax : float
        Maximum value for sodium conductance (uS)
    gkdmin : float
        Minimum value for potassium conductance (uS)
    gkdmax : float
        Maximum value for potassium conductance (uS)

    Returns
    -------
    traces : brian2.units.fundamentalunits.Quantity
        Arrays of simulated membrane voltage traces returned by the model (mV)

    Notes
    -----
    The time step is set at dt = 0.02 ms
    """
    
    # area = 20000*umetre**2
    # Cm=1*ufarad*cm**-2 * area
    Cm = cm_var*pF
    El=el*mV
    EK=ek*mV
    ENa=ena*mV
    VT=vt*mV


    model = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    g_na : siemens (constant)
    g_kd : siemens (constant)
    gl   : siemens (constant)
    '''

    opt = NevergradOptimizer()
    metric = MSEMetric()

    fitter = TraceFitter(model=model,
                        input_var='I',
                        output_var='v',
                        input=input_current * amp,
                        output=output_response * mV,
                        dt=0.02 * ms,
                        n_samples=100,
                        method='exponential_euler',
                        param_init={'v': El})

    np.float = float

    print('first fit:')
    res, error = fitter.fit(n_rounds=round_num,
                            optimizer=opt,
                            metric=metric,
                            gl=[glmin*nsiemens, glmax*nsiemens],
                            g_na=[gnamin*usiemens, gnamax*usiemens],
                            g_kd=[gkdmin*usiemens, gkdmax*usiemens])

    traces = fitter.generate_traces()

    return traces


def run_hh_all_params(input_current, output_response, round_num=10, cm_min=80, cm_max=300, 
                      el_min=-90, el_max=-50, ek_min=-90, ek_max=-40, ena_min=-40, ena_max=40, 
                      vt_min=-60, vt_max=-40, glmin=10, glmax=40, gnamin=1, gnamax=20, gkdmin=0.01, gkdmax=4):
    """
    Run the Hodgkin-Huxley model on all parameters.

    Parameters
    ----------
    input_current : array_like
        Array of input values.
    output_response : array_like
        Array of output values.
    round_num : int
        Number of rounds
    cm_min : float
        Minimum value for capacitance (pF)
    cm_max : float
        Maximum value for capacitance (pF)
    el_min : float
        Minimum value for leak reversal potential (mV)
    el_max : float
        Maximum value for leak reversal potential (mV)
    ek_min : float
        Minimum value for potassium reversal potential (mV)
    ek_max : float
        Maximum value for potassium reversal potential (mV)
    ena_min : float
        Minimum value for sodium reversal potential (mV)
    ena_max : float
        Maximum value for sodium reversal potential (mV)
    vt_min : float
        Minimum value for spike threshold (mV)
    vt_max : float
        Maximum value for spike threshold (mV)
    glmin : float
        Minimum value for leak conductance (nS)
    glmax : float
        Maximum value for leak conductance (nS)
    gnamin : float
        Minimum value for sodium conductance (uS)
    gnamax : float
        Maximum value for sodium conductance (uS)
    gkdmin : float
        Minimum value for potassium conductance (uS)
    gkdmax : float
        Maximum value for potassium conductance (uS)

    Returns
    -------
    traces : brian2.units.fundamentalunits.Quantity
        Arrays of simulated membrane voltage traces returned by the model (mV)

    Notes
    -----
    The time step is set at dt = 0.02 ms
    """

    model = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    Cm : farad (constant)
    El : volt (constant)
    EK : volt (constant)
    ENa : volt (constant)
    VT : volt (constant)    
    g_na : siemens (constant)
    g_kd : siemens (constant)
    gl   : siemens (constant)

    '''

    opt = NevergradOptimizer()
    metric = MSEMetric()

    fitter = TraceFitter(model=model,
                        input_var='I',
                        output_var='v',
                        input=input_current * amp,
                        output=output_response * mV,
                        dt=0.02 * ms,
                        n_samples=100,
                        method='exponential_euler',
                        param_init={'v': -70*mV})

    np.float = float

    print('first fit:')
    res, error = fitter.fit(n_rounds=round_num,
                            optimizer=opt,
                            metric=metric,
                            Cm=[cm_min*pF, cm_max*pF],
                            El=[el_min*mV, el_max*mV],
                            EK=[ek_min*mV, ek_max*mV],
                            ENa=[ena_min*mV, ena_max*mV],
                            VT=[vt_min*mV, vt_max*mV],
                            gl=[glmin*nsiemens, glmax*nsiemens],
                            g_na=[gnamin*usiemens, gnamax*usiemens],
                            g_kd=[gkdmin*usiemens, gkdmax*usiemens])

    traces = fitter.generate_traces()

    return traces


def run_lif(in_c, out_r, el=-70, reset_var='v=-50*mV', thresh_var= 'v>-40*mV', refrac=0.1, \
            round_num=10, r_min=100*10**6, r_max=300*10**6, t_min=0.01, t_max=0.03):
    """
    Run the Leaky Integrate-and-Fire model.

    Parameters
    ----------
    in_c : array_like
        Array of input current values (unitless - converted to amps internally).
    out_r : array_like
        Array of output voltage values (unitless - converted to mV internally).
    el : float
        Numeric value for leak reversal potential (mV)
    reset_var : str
        String representing the set value for the reset variable
    thresh_var : str
        String representing the set value for the threshold variable
    refrac : float
        Numeric value for refractory period (ms)
    round_num : int
        Number of rounds
    r_min : int
        Minimum value for resistance (ohm)
    r_max : int
        Maximum value for resistance (ohm)
    t_min : float
        Minimum value for tau (second)
    t_max : float
        Maximum value for tau (second)
        
    Returns
    -------
    traces : brian2.units.fundamentalunits.Quantity
        Arrays of simulated membrane voltage traces returned by the model (mV)

    Notes
    -----
    The time step is set at dt = 0.02 ms
    """
    El=el*mV

    model = '''
    dv/dt = (-(v-El)+R*I)/tau : volt #(unless refractory)
    R : ohm (constant)
    tau : second (constant)
    '''

    opt = NevergradOptimizer()
    metric = MSEMetric()

    fitter = TraceFitter(model=model,
                        input_var='I',
                        output_var='v',
                        input = in_c * amp,
                        output = out_r * mV,
                        dt = 0.02 * ms,
                        n_samples=100,
                        method='exponential_euler',
                        reset=reset_var,
                        threshold=thresh_var,
                        refractory=refrac*ms,
                        param_init={'v': El})

    np.float = float

    res, error = fitter.fit(n_rounds=round_num,
                            optimizer=opt,
                            metric=metric,
                            R=[r_min*ohm, r_max*ohm],
                            tau=[t_min*second, t_max*second])

    traces = fitter.generate_traces()

    return traces


def run_adex(input_data, output_data, c_var=120, gl_var=5, el=-70, vt_var=-50, delt_var=2, vc_var=5, 
            vr=-70, round_num=10, tauw_min=120, tauw_max=160, a_min=1, a_max=4, b_min=0, b_max=0.5):
    """
    Run the Adaptive Exponential Integrate-and-Fire model.

    Parameters
    ----------
    input_data : array_like
        Array of input current values (unitless - converted to amps internally).
    output_data : array_like
        Array of output voltage values (unitless - converted to mV internally).
    c_var : float
        Numeric value for capacitance (pF).
    gl_var : float
        Numeric value for leak conductance (nS).
    el : float
        Numeric value for leak reversal potential (mV).
    vt_var : float
        Numeric value for soft voltage spike initiation (mV).
    delt_var : float
        Numeric value for the exponential slope factor (mV).
    vc_var : float
        Numeric value which, with VT and DeltaT, affects Vcut.
    vr : float
        Numeric value for the reset membrane potential (mV).
    round_num : int
        Number of rounds.
    tauw_min : int
        Minimum value for adaptation time constant (ms).
    tauw_max : int
        Maximum value for adaptation time constant (ms).
    a_min : float
        Minimum value for subthreshold adaptation strength (nS).
    a_max : float
        Maximum value for subthreshold adaptation strength (nS).
    b_min : float
        Minimum value for spike-triggered adaptation (nA).
    b_max : float
        Maximum value for spike-triggered adaptation (nA).

    Returns
    -------
    traces : brian2.units.fundamentalunits.Quantity
        Arrays of simulated membrane voltage traces returned by the model (mV)

    Notes
    -----
    The time step is set at dt = 0.02 ms
    """
    # Parameters
    C = c_var * pF
    gL = gl_var * nS
    # taum = C / gL
    El = el * mV
    VT = vt_var * mV
    DeltaT = delt_var * mV
    Vcut = VT + vc_var * DeltaT
    Vr = vr*mV

    model = """
    dv/dt = (gL*(El - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - w)/C : volt
    dw/dt = (a*(v - El) - w)/tauw : amp

    tauw : second (constant)
    a : siemens (constant)
    b : amp (constant)
    """

    opt = NevergradOptimizer()
    metric = MSEMetric()

    fitter = TraceFitter(model=model,
                        input_var='I',
                        output_var='v',
                        input = input_data * amp,
                        output = output_data * mV,
                        dt= 0.02 * ms,
                        n_samples=100,
                        method='exponential_euler',
                        reset='v=Vr; w+=b',
                        threshold='v>Vcut',
                        # refractory=refrac*ms,
                        param_init={'v': El})

    np.float = float

    print('first fit:')
    res, error = fitter.fit(n_rounds=round_num,
                            optimizer=opt,
                            metric=metric,
                            tauw=[tauw_min*ms, tauw_max*ms],
                            a=[a_min*nsiemens, a_max*nsiemens],
                            b=[b_min*nA, b_max*nA])

    traces = fitter.generate_traces()

    return traces


#####################################################################################################################
# PLOTS
#####################################################################################################################


def create_initial_in_out(inpt, outpt, initial_ind):
    """
    Generates a grid of 2x5 subplots showing input currents and output responses.

    Parameters
    ----------
    inpt : array_like
        Array of input values.
    outpt : array_like
        Array of output.
    initial_ind : int
        initial indice number.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.

    """

    figure(figsize=(12,4))

    ts = (np.arange(0, len(inpt[0]))) * (1/50)

    subplot(2,5,1)
    plot(ts, inpt[initial_ind])
    tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
    ylabel('membrane potential (mV)')

    subplot(2,5,2)
    plot(ts, inpt[initial_ind+1])
    tick_params(
    axis='x',          
    which='both',     
    bottom=False,      
    top=False,         
    labelbottom=False)
    yticks([])

    subplot(2,5,3)
    plot(ts, inpt[initial_ind+2])
    title(f'Input current and output response for sweeps {initial_ind} - {initial_ind+4}')
    tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
    yticks([])

    subplot(2,5,4)
    plot(ts, inpt[initial_ind+3])
    tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
    yticks([])

    subplot(2,5,5)
    plot(ts, inpt[initial_ind+4])
    tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
    yticks([])

    subplot(2,5,6)
    plot(ts, outpt[initial_ind])
    xlabel('time (ms)')
    ylabel('input current (pA)')

    subplot(2,5,7)
    plot(ts, outpt[initial_ind+1])
    yticks([])

    subplot(2,5,8)
    plot(ts, outpt[initial_ind+2])
    yticks([])

    subplot(2,5,9)
    plot(ts, outpt[initial_ind+3])
    yticks([])

    subplot(2,5,10)
    plot(ts, outpt[initial_ind+4])
    yticks([])


def out_in_vertical(d1, d2):
    """
    Generates a grid of 2x1 subplots showing input currents and output responses.

    Parameters
    ----------
    d1 : array_like
        Array of first dataset.
    d2 : array_like
        Array of second dataset.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.

    """

    time_step = time_steps(d1)

    subplot(2,1,1)
    plot(time_step, d1, label='output')
    tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
    ylabel('membrane potential (mV)')
    legend()

    subplot(2,1,2)
    plot(time_step, d2, label='input')
    xlabel('time (ms)')
    ylabel('input current (pA)')
    legend()


def plot_vertical_fit(d1, d2, d1_label='output', d2_label='input', t_name='time (ms)', \
                      xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None):
    """
    Generates a 2x1 grid of subplots showing a visual comparison of outputs over to time.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    d1_label : str
        Label for dataset 1.
    d2_label : str
        Label for dataset 2.
    t_name : str
        Label for time (i.e. seconds or milliseconds).
    xlimmin : None or float
        Minimum value for the x-axis.
    xlimmax : None or float
        Maximum value for the x-axis.
    ylimmin : None or float
        Minimum value for the y-axis.
    ylimmax : None or float
        Maximum value for the y-axis.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.

    """

    times = time_steps(d1)

    subplot(2,1,1)
    plot(times, d1, label=d1_label)
    tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)
    ylabel('membrane potential (mV)')
    xlim(xlimmin, xlimmax)
    ylim(ylimmin, ylimmax)
    legend()

    subplot(2,1,2)
    plot(times, d2, label=d2_label)
    xlabel(t_name)
    ylabel('input current (pA)')
    xlim(xlimmin, xlimmax)
    ylim(ylimmin, ylimmax)
    legend()


def create_comparison_together(d1, d2, d1_title='target', d2_title='fit'):
    """
    Generates a plot showing a visual comparison of two outputs overlaid.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    d1_title : str
        Label for dataset 1.
    d2_title : str
        Label for dataset 2.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """

    time_axis = time_steps(d1)
    plot(time_axis, d1, label = d1_title)
    plot(time_axis, d2, label = d2_title)
    xlabel('time (ms)')
    ylabel('membrane potential (mV)')
    legend()


def create_comparison_together_three(d1, d2, d3, d1_title='target', d2_title='fit 1', d3_title='fit 2'):
    """
    Generates a plot showing a visual comparison of three outputs overlaid.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    d3 : array_like
        Array of values for dataset 3.
    d1_title : str
        Label for dataset 1.
    d2_title : str
        Label for dataset 2.
    d3_title : str
        Label for dataset 3.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """

    time_axis = time_steps(d1)
    plot(time_axis, d1, label = d1_title)
    plot(time_axis, d2, label = d2_title)
    plot(time_axis, d3, label = d3_title)
    xlabel('time (ms)')
    ylabel('membrane potential (mV)')
    legend()


def create_comparison_together_four(d1, d2, d3, d4, d1_title='target', d2_title='fit 1', 
                                    d3_title='fit 2', d4_title='fit 3'):
    """
    Generates a plot showing a visual comparison of four outputs overlayed.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    d3 : array_like
        Array of values for dataset 3.
    d4 : array_like
        Array of values for dataset 4.
    d1_title : str
        Label for dataset 1.
    d2_title : str
        Label for dataset 2.
    d3_title : str
        Label for dataset 3.
    d4_title : str
        Label for dataset 4.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """

    time_axis = time_steps(d1)
    plot(time_axis, d1, label = d1_title)
    plot(time_axis, d2, label = d2_title)
    plot(time_axis, d3, label = d3_title)
    plot(time_axis, d4, label = d4_title)
    xlabel('time (ms)')
    ylabel('membrane potential (mV)')
    legend()


def create_comparison_across_two(d1, d2, ind1=0, ind2=1, t1='', t2='', title_of_whole=''):
    """
    Generates a grid of 1x2 subplots showing a horizontal comparison between outputs.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    ind1 : int
        Index for subplot 1.
    ind2 : int
        Index for subplot 2.
    t1 : str
        Title for subplot 1.
    t2 : str
        Title for subplot 2.
    title_of_whole : str
        Title for the whole plot.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """
    figure(figsize=(12,4))

    subplot(1,2,1)
    create_comparison_together(d1[ind1], d2[ind1]*1000, d2_title=t1)
    ylabel('membrane potential (mV)')
    title(title_of_whole)

    subplot(1,2,2)
    create_comparison_together(d1[ind2], d2[ind2]*1000, d2_title=t2)
    ax = plt.gca()
    ax.yaxis.set_visible(False)


def create_comparison_across_three(d1, d2, ind=0, t1='', t2='', t3='', title_of_whole=''):
    """
    Generates a grid of 1x3 subplots showing a horizontal comparison between outputs.

    Parameters
    ----------
    d1 : array_like
        Array of values for dataset 1.
    d2 : array_like
        Array of values for dataset 2.
    ind : int
        Index for arrays.
    t1 : str
        Title for subplot 1.
    t2 : str
        Title for subplot 2.
    t3 : str
        Title for subplot 3.
    title_of_whole : str
        Title for the whole plot.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """
    figure(figsize=(12,4))

    subplot(1,3,1)
    create_comparison_together(d1[ind], d2[ind]*1000, d2_title=t1)
    ylabel('membrane potential (mV)')

    subplot(1,3,2)
    create_comparison_together(d1[ind+1], d2[ind+1]*1000, d2_title=t2)
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    title(title_of_whole)

    subplot(1,3,3)
    create_comparison_together(d1[ind+2], d2[ind+2]*1000, d2_title=t3)
    ax = plt.gca()
    ax.yaxis.set_visible(False)


def one_plot(data):
    """
    Generates a plot showing membrane potential over time.

    Parameters
    ----------
    data : array_like
        Array of values for dataset.

    Returns
    -------
    None
        This function creates and displays a Matplotlib figure.
    """

    time_axis = time_steps(data)
    plot(time_axis, data)
    xlabel('time (ms)')
    ylabel('membrane potential (mV)')
