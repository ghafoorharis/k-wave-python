""" This module contains the functions to generate the phantom data for the US B-mode linear transducer example.
"""

import logging
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

# from examples.us_bmode_linear_transducer.example_utils import download_if_does_not_exist
from kwave.data import Vector  # type: ignore
from kwave.utils.mapgen import make_ball  # type: ignore


def get_phantom_data_circle(
    kgrid,
    params,
    debug=True,
):
    """This method returns a phantom with a circle object.
    Args:
        kgrid (dict): _description_
        params (dict): _description_
        debug (bool): _description_
    Returns:
        dict: returns a phantom with a circle object wrapped in a dictionary
        with the following keys: sound_speed_map, density_map
    """
    # Load the parameters
    background_map_mean = params["background_map_mean"]
    background_map_std = params["background_map_std"]
    object_map_mean = params["object_map_mean"]
    object_map_std = params["object_map_std"]
    object_map_radius = params["object_map_radius"]
    object_map_center_x = params["object_map_center_x"]
    object_map_center_y = params["object_map_center_y"]
    object_map_center_z = params["object_map_center_z"]
    density_constant_multiplier = params["density_constant_multiplier"]
    lower_amplitude = params["lower_amplitude"]
    upper_amplitude = params["upper_amplitude"]
    c0 = params["c0"]
    rho0 = params["rho0"]
    grid_size_points = params["grid_size_points"]

    # Define the properties of the propagation medium
    background_map = background_map_mean + background_map_std * np.random.randn(
        kgrid.Nx, kgrid.Ny, kgrid.Nz
    )
    sound_speed_map = c0 * background_map
    density_map = rho0 * background_map
    if debug:
        init_sound_speed_map = sound_speed_map.copy()
        init_density_map = density_map.copy()
    # Define a random distribution of scatterers for the highly scattering region
    scattering_map = np.random.randn(kgrid.Nx, kgrid.Ny, kgrid.Nz)
    scattering_c0 = np.clip(
        c0 + object_map_mean + object_map_std * scattering_map,
        lower_amplitude,
        upper_amplitude,
    )
    scattering_rho0 = scattering_c0 / density_constant_multiplier
    # Define a sphere for a highly scattering region
    scattering_region = get_3d_circle(
        kgrid=kgrid,
        grid_size_points=grid_size_points,
        radius=object_map_radius,
        x_pos=object_map_center_x,
        y_nod=object_map_center_y,
        z_nod=object_map_center_z,
    )
    # Update the sound speed and density map
    # all the locations where the scattering region is not zero, the speed sound map & density map will be updated with highly scattering region sound speed
    # otherwise, all other locations are updated with the background sound speed
    sound_speed_map[scattering_region] = scattering_c0[scattering_region]
    density_map[scattering_region] = scattering_rho0[scattering_region]
    if debug:
        visualize_maps(
            sound_speed_map=init_sound_speed_map,
            density_map=init_density_map,
            scattering_map=scattering_map,
            scattering_c0=scattering_c0,
            scattering_rho0=scattering_rho0,
            scattering_region=scattering_region,
            final_sound_speed_map=sound_speed_map,
            final_density_map=density_map,
            save_path=f"{params['RESULTS_DIR']}/{params['experiment_name']}/phantom_maps.png",
            show_fig=params["show_fig"],
            params=params,
        )
    # Return
    phantom = {"sound_speed_map": sound_speed_map, "density_map": density_map}
    return phantom


def get_3d_circle(kgrid, grid_size_points, radius=8e-3, x_pos=32e-3, y_nod=2, z_nod=2):
    """
    This method returns a 3D circle object.
    Args:
        kgrid (dict): _description_
        grid_size_points (dict): _description_
        radius (float): _description_
        x_pos (float): _description_
        y_nod (int): _description_
        z_nod (int): _description_
    """
    # Define a sphere for a highly scattering region
    y_pos = kgrid.dy * kgrid.Ny / y_nod
    z_pos = kgrid.dz * kgrid.Nz / z_nod
    ball_center = np.round(Vector([x_pos, y_pos, z_pos]) / kgrid.dx)
    scattering_region = make_ball(
        grid_size_points, ball_center, round(radius / kgrid.dx)
    ).nonzero()
    return scattering_region


def visualize_maps(
    sound_speed_map,
    scattering_map,
    density_map,
    scattering_c0,
    scattering_rho0,
    scattering_region,
    save_path,
    final_sound_speed_map=None,
    final_density_map=None,
    show_fig=False,
    params=None,
):
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    # multiple subplots
    # Visualize the sound speed map
    axs[0, 0].imshow(sound_speed_map[:, :, 64], cmap="gray")
    # Visualize the sound speed map
    axs[0, 0].set_title("Sound Speed Map")

    # Visualize the density map
    axs[0, 1].imshow(density_map[:, :, 64], cmap="gray")
    axs[0, 1].set_title("Density Map")

    # Visualize the scattering map
    axs[0, 2].imshow(scattering_map[:, :, 64], cmap="gray")
    axs[0, 2].set_title("Scattering Map")

    # Visualize the scattering c0 map
    axs[1, 0].imshow(scattering_c0[:, :, 64], cmap="gray")
    axs[1, 0].set_title("Scattering c0")

    # Visualize the scattering rho0 map
    axs[1, 1].imshow(scattering_rho0[:, :, 64], cmap="gray")
    axs[1, 1].set_title("Scattering rho0")
    # Visualize the scattering region
    scattering_region_map = np.zeros_like(sound_speed_map)
    scattering_region_map[scattering_region] = 1
    axs[1, 2].imshow(scattering_region_map[:, :, 64], cmap="gray")
    axs[1, 2].set_title("Scattering Region")
    # Visualize the final sound speed map
    axs[2, 1].imshow(final_sound_speed_map[:, :, 64], cmap="gray")
    axs[2, 1].set_title("Final Sound Speed Map")
    # Visualize the final density map
    axs[2, 2].imshow(final_density_map[:, :, 64], cmap="gray")
    axs[2, 2].set_title("Final Density Map")
    # Adjust layout
    plt.tight_layout()
    fig.savefig(save_path)
    if show_fig:
        plt.show()
    else:
        plt.close() 
    # # Show the figure
    # plt.show()


def visualize_receiver_part(
    kgrid,
    sound_speed_map,
    scan_lines_no_input,
    scan_lines_tgc,
    scan_lines_fund_ex,
    scan_lines_fund_env_ex,
    scan_lines_fund_log_ex,
    scan_lines_fund,
    scan_lines_harm,
    grid_size_points,
    c0,
    t_end,
    args
):
    """
    This method visualizes the receiver part of the simulation.
    Args:
        kgrid (dict): _description_
        sound_speed_map (np.array): _description_
        scan_lines_no_input (np.array): _description_
        scan_lines_tgc (np.array): _description_
        scan_lines_fund_ex (np.array): _description_
        scan_lines_fund_env_ex (np.array): _description_
        scan_lines_fund_log_ex (np.array): _description_
        scan_lines_fund (np.array): _description_
        scan_lines_harm (np.array): _description_
        grid_size_points (dict): _description_
        c0 (float): _description_
        t_end (float): _description_

    """
    # Set the desired size of the image
    image_size = kgrid.size

    # Create the axis variables
    x_axis = [0, image_size[0] * 1e3 * 1.1]  # [mm]
    y_axis = [-0.5 * image_size[1] * 1e3, 0.5 * image_size[1] * 1e3]  # [mm]

    # make plotting non-blocking
    plt.ion()
    # Plot the data before and after scan conversion
    plt.figure(figsize=(14, 4))
    # plot the sound speed map
    plt.subplot(1, 3, 1)
    plt.imshow(
        sound_speed_map[:, 64:-64, int(grid_size_points.z / 2)],
        aspect="auto",
        extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
    )
    plt.title("Sound Speed")
    plt.xlabel("Width [mm]")
    plt.ylabel("Depth [mm]")
    ax = plt.gca()
    ax.set_ylim(40, 5)
    plt.subplot(1, 3, 2)
    plt.imshow(
        scan_lines_fund.T,
        cmap="grey",
        aspect="auto",
        extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
    )
    plt.xlabel("Width [mm]")
    plt.title("Fundamental")
    ax = plt.gca()
    ax.set_ylim(40, 5)
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(
        scan_lines_harm.T,
        cmap="grey",
        aspect="auto",
        extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
    )
    plt.yticks([])
    plt.xlabel("Width [mm]")
    plt.title("Harmonic")
    ax = plt.gca()
    ax.set_ylim(40, 5)
    plt.tight_layout()
    plt.savefig(f"{args['RESULTS_DIR']}/exp_{args['experiment_name']}/visualization_maps.png")

    # Display the plots for 120 seconds

    # Creating a dictionary with the step labels as keys
    processing_steps = {
        "1. Beamformed Signal": scan_lines_no_input,
        "2. Time Gain Compensation": scan_lines_tgc,
        "3. Frequency Filtering": scan_lines_fund_ex,
        "4. Envelope Detection": scan_lines_fund_env_ex,
        "5. Log Compression": scan_lines_fund_log_ex,
    }

    plt.figure(figsize=(14, 4), tight_layout=True)

    offset = -6e5
    # Plotting each step using the dictionary
    for i, (label, data) in enumerate(processing_steps.items()):
        plt.plot(kgrid.t_array.squeeze(), data.squeeze() + offset * i, label=label)

    # Set y-ticks and y-labels
    plt.yticks([offset * i for i in range(5)], list(processing_steps.keys()))
    plt.xlabel("Time [\u03BCs]")
    plt.xlim(5e-3 * 2 / c0, t_end)
    plt.title("Processing Steps Visualization")
    # plt.pause(120)  # Display plots for 120 seconds for CI/CD to complete
    plt.savefig(f"{args['RESULTS_DIR']}/exp_{args['experiment_name']}/rf_processing_result.png")


# if __name__ == "__main__":
#     import logging
#     import numpy as np
#     import scipy.io
#     import matplotlib.pyplot as plt


#     # from examples.us_bmode_linear_transducer.example_utils import download_if_does_not_exist
#     from kwave.data import Vector
#     from kwave.kgrid import kWaveGrid
#     from kwave.kmedium import kWaveMedium
#     from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
#     from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
#     from kwave.options.simulation_execution_options import SimulationExecutionOptions
#     from kwave.options.simulation_options import SimulationOptions
#     from kwave.utils.dotdictionary import dotdict
#     from kwave.utils.signals import tone_burst, get_win
#     from kwave.utils.filters import gaussian_filter
#     from kwave.utils.conversion import db2neper
#     from kwave.reconstruction.tools import log_compression
#     from kwave.reconstruction.beamform import envelope_detection
#     from kwave.utils.mapgen import make_ball
#     from us_bmode_linear_transducer.utils_ex.artifacts import get_phantom_data_circle, get_3d_circle,visualize_maps
#     from us_bmode_linear_transducer.utils_ex.sim_params import Params
#     args = Params()
#     grid_size_points = args.grid_size_points
#     grid_spacing_meters = args.grid_spacing_meters
#     c0 = args.c0
#     rho0 = args.rho0
#     tone_burst_freq = args.tone_burst_freq
#     tone_burst_cycles = args.tone_burst_cycles
#     SOME_TIME_CONSTANT = args.SOME_TIME_CONSTANT
#     source_strength = args.source_strength
#     kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
#     t_end = (grid_size_points.x * grid_spacing_meters.x) * SOME_TIME_CONSTANT / c0  # [s]
#     kgrid.makeTime(c0, t_end=t_end)

#     input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
#     input_signal = (source_strength / (c0 * rho0)) * input_signal

#     medium = kWaveMedium(
#         sound_speed=None,  # will be set later
#         alpha_coeff=args.alpha_coeff,
#         alpha_power=args.alpha_power,
#         BonA=args.BonA,
#     )

#     transducer = dotdict()
#     transducer.number_elements = 32  # total number of transducer elements
#     transducer.element_width = 2  # width of each element [grid points/voxels]
#     transducer.element_length = 24  # length of each element [grid points/voxels]
#     transducer.element_spacing = (
#         0  # spacing (kerf  width) between the elements [grid points/voxels]
#     )
#     transducer.radius = float("inf")  # radius of curvature of the transducer [m]

#     # calculate the width of the transducer in grid points
#     transducer_width = (
#         transducer.number_elements * transducer.element_width
#         + (transducer.number_elements - 1) * transducer.element_spacing
#     )

#     # use this to position the transducer in the middle of the computational grid
#     transducer.position = np.round(
#         [
#             1,
#             grid_size_points.y / 2 - transducer_width / 2,
#             grid_size_points.z / 2 - transducer.element_length / 2,
#         ]
#     )
#     transducer = kWaveTransducerSimple(kgrid, **transducer)


#     not_transducer = dotdict()
#     not_transducer.sound_speed = c0  # sound speed [m/s]
#     not_transducer.focus_distance = 20e-3  # focus distance [m]
#     not_transducer.elevation_focus_distance = (
#         19e-3  # focus distance in the elevation plane [m]
#     )
#     not_transducer.steering_angle = 0  # steering angle [degrees]
#     not_transducer.transmit_apodization = "Hanning"
#     not_transducer.receive_apodization = "Rectangular"
#     not_transducer.active_elements = np.ones((transducer.number_elements, 1))
#     not_transducer.input_signal = input_signal

#     not_transducer = NotATransducer(transducer, kgrid, **not_transducer)
#     logging.log(logging.INFO, "Fetching phantom data...")
#     # download_if_does_not_exist(PHANTOM_DATA_GDRIVE_ID, PHANTOM_DATA_PATH)
#     params_dict = args.__dict__
#     params_dict["experiment_name"] = "debug_exp"
#     print(params_dict)
#     phantom = get_phantom_data_circle(
#                                     kgrid=kgrid,
#                                     params = params_dict,
#                                     debug = True
#                                     )

#     sound_speed_map = phantom["sound_speed_map"]
#     density_map = phantom["density_map"]
