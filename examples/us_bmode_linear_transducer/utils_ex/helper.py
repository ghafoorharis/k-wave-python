import logging
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

# external imports

from kwave.data import Vector  # type: ignore
from kwave.kgrid import kWaveGrid  # type: ignore
from kwave.kmedium import kWaveMedium  # type: ignore
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D  # type: ignore
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple  # type: ignore
from kwave.options.simulation_execution_options import SimulationExecutionOptions  # type: ignore
from kwave.options.simulation_options import SimulationOptions  # type: ignore
from kwave.utils.dotdictionary import dotdict  # type: ignore
from kwave.utils.signals import tone_burst, get_win  # type: ignore
from kwave.utils.filters import gaussian_filter  # type: ignore
from kwave.utils.conversion import db2neper  # type: ignore
from kwave.reconstruction.tools import log_compression  # type: ignore
from kwave.reconstruction.beamform import envelope_detection  # type: ignore
from kwave.utils.mapgen import make_ball  # type: ignore

# internal imports
from utils_ex.artifacts import visualize_receiver_part,plot_data_using_phaser_template

# create the computational grid
def make_grid(grid_size_points, grid_spacing_meters, c0, t_end):
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
    kgrid.makeTime(c0, t_end=t_end)
    return kgrid


# create the time array and input signal
def get_input_signal(
    kgrid, tone_burst_freq, tone_burst_cycles, source_strength, c0, rho0
):
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (c0 * rho0)) * input_signal
    return input_signal


# define the transducer properties
def get_transducer(grid_size_points, input_signal, kgrid, c0,param_elem_width  = 2,sc = 1):

    transducer = dotdict()
    transducer.number_elements = 32 /sc # total number of transducer elements
    transducer.element_width = param_elem_width  # width of each element [grid points/voxels]
    transducer.element_length = 24 /sc  # length of each element [grid points/voxels]
    transducer.element_spacing = (
        0  # spacing (kerf  width) between the elements [grid points/voxels]
    )
    transducer.radius = float("inf")  # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = (
        transducer.number_elements * transducer.element_width
        + (transducer.number_elements - 1) * transducer.element_spacing
    )

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round(
        [
            1,
            grid_size_points.y / 2 - transducer_width / 2,
            grid_size_points.z / 2 - transducer.element_length / 2,
        ]
    )
    transducer = kWaveTransducerSimple(kgrid, **transducer)

    not_transducer = dotdict()
    not_transducer.sound_speed = c0  # sound speed [m/s]
    not_transducer.focus_distance = 20e-3  # focus distance [m]
    not_transducer.elevation_focus_distance = (
        19e-3  # focus distance in the elevation plane [m]
    )
    not_transducer.steering_angle = 0  # steering angle [degrees]
    not_transducer.transmit_apodization = "Hanning"
    not_transducer.receive_apodization = "Rectangular"
    not_transducer.active_elements = np.ones((transducer.number_elements, 1))
    not_transducer.input_signal = input_signal

    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

    return not_transducer, transducer


def solve_kspace_problem(
    kgrid,
    medium,
    args,
    sound_speed_map,
    density_map,
    grid_size_points,
    transducer,
    not_transducer,
) -> np.array:
    """This function solves the kspace problem for the given parameters
    Args:
        kgrid (object): kWaveGrid object
        medium (object): kWaveMedium object
        args (object): Params object
        sound_speed_map (np.array): sound speed map
        density_map (np.array): density map
        grid_size_points (object): Vector
        transducer (object): kWaveTransducerSimple object
        not_transducer (object): NotATransducer object
    Returns:
        scan lines: np.array
    """
    # preallocate the storage set medium position
    scan_lines = np.zeros((args.number_scan_lines, kgrid.Nt))
    medium_position = 0

    for scan_line_index in range(0, args.number_scan_lines):
        print(f"Computing scan line {scan_line_index} of {args.number_scan_lines}")

        # load the current section of the medium
        medium.sound_speed = sound_speed_map[
            :, medium_position : medium_position + grid_size_points.y, :
        ]
        medium.density = density_map[
            :, medium_position : medium_position + grid_size_points.y, :
        ]

        # set the input settings
        input_filename = f"example_input_{scan_line_index}.h5"
        # set the input settings
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=args.pml_size_points,
            data_cast=args.DATA_CAST,
            data_recast=True,
            save_to_disk=True,
            input_filename=input_filename,
            save_to_disk_exit=False,
        )
        # run the simulation
        if args.RUN_SIMULATION:
            sensor_data = kspaceFirstOrder3D(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=not_transducer,
                simulation_options=simulation_options,
                execution_options=SimulationExecutionOptions(is_gpu_simulation=True),
            )

            scan_lines[scan_line_index, :] = not_transducer.scan_line(
                not_transducer.combine_sensor_data(sensor_data["p"].T)
            )

        # update medium position
        medium_position = medium_position + transducer.element_width

    RESULTS_DIR = args.RESULTS_DIR
    EXP_NAME = args.experiment_name
    SAVE_PATH = f"{RESULTS_DIR}/{EXP_NAME}/phantom_data.mat"
    if args.RUN_SIMULATION:
        simulation_data = scan_lines
        scipy.io.savemat(SAVE_PATH, {"scan_lines": simulation_data})

    else:
        logging.log(logging.INFO, "loading data from local disk...")
        # download_if_does_not_exist(SENSOR_DATA_GDRIVE_ID, sensor_data_path)
        simulation_data = scipy.io.loadmat(SAVE_PATH)["scan_lines"]

    return simulation_data


def process_simulation_data(
    kgrid, medium, input_signal, scan_lines, tone_burst_freq, c0, args,
    t_end
):
    """This function contains the processoing of scan lines that results in the b-mode image
    Args:
        None
    Returns:
        B-mode image: np.array
    """
    debug = args["debug"]
    SHIFTING_SAMP_FREQ = args["SHIFTING_SAMP_FREQ"]
    BW_FUND = args["BW_FUND"]
    BW_HARM = args["BW_HARM"]

    # trim the offset delay and remove the input signal from the scan lines
    scan_lines, scan_lines_no_input = trim_offset_delay(
        kgrid=kgrid, input_signal=input_signal, tukey_win=None, scan_lines=scan_lines
    )
    # apply time gain compensation
    scan_lines, scan_lines_tgc, r = time_gain_compensation(
        input_signal=input_signal,
        scan_lines=scan_lines,
        kgrid=kgrid,
        medium=medium,
        tone_burst_freq=tone_burst_freq,
        c0=c0,
        Nt=kgrid.Nt,
    )
    # apply filtering
    scan_lines_fund, scan_lines_harm, scan_lines_fund_ex, scan_lines_harm_ex = (
        filtering(
            scan_lines=scan_lines,
            kgrid=kgrid,
            tone_burst_freq=tone_burst_freq,
            SHIFTING_SAMP_FREQ=SHIFTING_SAMP_FREQ,
            BW_FUND=BW_FUND,
            BW_HARM=BW_HARM,
        )
    )
    # apply envelope detection
    scan_lines_fund, scan_lines_harm, scan_lines_fund_env_ex, scan_lines_harm_env_ex = (
        detect_env(
            scan_lines_fund=scan_lines_fund, scan_lines_harm=scan_lines_harm
        )
    )
    # apply log compression
    scan_lines_fund, scan_lines_harm, scan_lines_fund_log_ex, scan_lines_harm_log_ex = (
        compress_log(
            scan_lines_fund=scan_lines_fund,
            scan_lines_harm=scan_lines_harm,
            compression_ratio=args["COMPRESSION_RATIO"],
        )
    )
    # # Intepolate the scan lines to the grid size
    # from scipy.interpolate import interp2d

    # def interp2(x, y, z, xi, yi):
    #     f = interp2d(x, y, z, kind='linear')
    #     zi = f(xi, yi)
    #     return zi

    # # Example usage:
    # x = np.arange(1, kgrid.Nt + 1)
    # y = np.arange(1, args["number_scan_lines"] + 1)
    # xi = np.linspace(1, kgrid.Nt, kgrid.Nt)
    # yi = np.linspace(1, args["number_scan_lines"], args["number_scan_lines"] * 2)  # Upsampling

    # scan_lines_fund = interp2(x, y, scan_lines_fund, xi, yi)
    # scan_lines_harm = interp2(x, y, scan_lines_harm, xi, yi)

    if debug:
        visualize_receiver_part(kgrid=kgrid, sound_speed_map=medium.sound_speed,
                                scan_lines_no_input=scan_lines_no_input,
                                scan_lines_tgc=scan_lines_tgc,
                                scan_lines_fund_ex=scan_lines_fund_ex,
                                scan_lines_fund_env_ex=scan_lines_fund_env_ex,
                                scan_lines_fund_log_ex=scan_lines_fund_log_ex,
                                scan_lines_fund=scan_lines_fund,
                                scan_lines_harm=scan_lines_harm,
                                grid_size_points=args["grid_size_points"],
                                c0=args["c0"],
                                t_end=t_end,
                                args=args
                                )
        print("Processing using the newly defined func")
        plot_data_using_phaser_template(
            scan_lines=scan_lines,
            scan_lines_fund = scan_lines_fund,
            scan_lines_harm = scan_lines_harm,
            steering_angles=None,
            c0=c0,
            kgrid=kgrid,
            medium=medium,
            # args
              )
                                
    return (
        scan_lines_fund,
        scan_lines_harm,
        scan_lines_fund_ex,
        scan_lines_harm_ex,
        scan_lines_fund_env_ex,
        scan_lines_harm_env_ex,
        scan_lines_fund_log_ex,
        scan_lines_harm_log_ex,
        r
    )


def trim_offset_delay(kgrid, input_signal, tukey_win, scan_lines):
    # Trim the delay offset from the scan line data
    tukey_win, _ = get_win(kgrid.Nt * 2, "Tukey", False, 0.05)
    transmit_len = len(input_signal.squeeze())
    scan_line_win = np.concatenate(
        (
            np.zeros([1, transmit_len * 2]),
            tukey_win.T[:, : kgrid.Nt - transmit_len * 2],
        ),
        axis=1,
    )
    scan_lines = scan_lines * scan_line_win
    # Remove the input signal from the scan lines
    scan_lines_no_input = scan_lines[len(scan_lines) // 2, :]
    return (
        scan_lines,
        scan_lines_no_input,
    )


def time_gain_compensation(
    input_signal, scan_lines, kgrid, medium, tone_burst_freq, c0, Nt
):
    # Create radius variable
    Nt = kgrid.Nt
    t0 = len(input_signal) * kgrid.dt / 2
    r = c0 * (np.arange(1, Nt + 1) * kgrid.dt - t0) / 2

    # Define absorption value and convert to correct units
    tgc_alpha_db_cm = (
        medium.alpha_coeff * (tone_burst_freq * 1e-6) ** medium.alpha_power
    )
    tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100

    # Create time gain compensation function
    tgc = np.exp(tgc_alpha_np_m * 2 * r)

    # Apply the time gain compensation to each of the scan lines
    scan_lines *= tgc

    # store intermediate results
    scan_lines_tgc = scan_lines[len(scan_lines) // 2, :]
    return scan_lines, scan_lines_tgc,r


def filtering(scan_lines, kgrid, tone_burst_freq, SHIFTING_SAMP_FREQ, BW_FUND, BW_HARM):
    scan_lines_fund = gaussian_filter(
        scan_lines, 1 / kgrid.dt, SHIFTING_SAMP_FREQ * tone_burst_freq, BW_FUND
    )
    scan_lines_harm = gaussian_filter(
        scan_lines, 1 / kgrid.dt, 2 * SHIFTING_SAMP_FREQ * tone_burst_freq, BW_HARM
    )
    # store intermediate results
    scan_lines_fund_ex = scan_lines_fund[len(scan_lines_fund) // 2, :]
    scan_lines_harm_ex = scan_lines_harm[len(scan_lines_harm) // 2, :]
    return scan_lines_fund, scan_lines_harm, scan_lines_fund_ex, scan_lines_harm_ex


def detect_env(scan_lines_fund, scan_lines_harm):
    scan_lines_fund = envelope_detection(scan_lines_fund)
    scan_lines_harm = envelope_detection(scan_lines_harm)
    # store intermediate results
    scan_lines_fund_env_ex = scan_lines_fund[len(scan_lines_fund) // 2, :]
    scan_lines_harm_env_ex = scan_lines_harm[len(scan_lines_harm) // 2, :]
    return (
        scan_lines_fund,
        scan_lines_harm,
        scan_lines_fund_env_ex,
        scan_lines_harm_env_ex,
    )


def compress_log(scan_lines_fund,
                 scan_lines_harm,
                 compression_ratio=3):
    scan_lines_fund = log_compression(scan_lines_fund, compression_ratio, True)
    scan_lines_harm = log_compression(scan_lines_harm, compression_ratio, True)
    # store intermediate results
    scan_lines_fund_log_ex = scan_lines_fund[len(scan_lines_fund) // 2, :]
    scan_lines_harm_log_ex = scan_lines_harm[len(scan_lines_harm) // 2, :]
    return (
        scan_lines_fund,
        scan_lines_harm,
        scan_lines_fund_log_ex,
        scan_lines_harm_log_ex,
    )
