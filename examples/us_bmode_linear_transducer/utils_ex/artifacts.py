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
from kwave.kgrid import kWaveGrid  # type: ignore
from kwave.reconstruction.beamform import envelope_detection, scan_conversion

def get_phantom_data_circle(
    kgrid,
    params,
    debug=True,
    element_width=2,
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
    number_scan_lines = params["number_scan_lines"]
    # Grid spacing according to the example
    Nx_tot = kgrid.Nx
    Ny_tot = kgrid.Ny + number_scan_lines * element_width
    Nz_tot = kgrid.Nz
    # Define the properties of the propagation medium
    background_map = background_map_mean + background_map_std * np.random.randn(
        Nx_tot, Ny_tot, Nz_tot
    )

    sound_speed_map = c0 * background_map
    density_map = rho0 * background_map
    if debug:
        init_sound_speed_map = sound_speed_map.copy()
        init_density_map = density_map.copy()
    # Define a random distribution of scatterers for the highly scattering region
    scattering_map = np.random.randn(Nx_tot, Ny_tot, Nz_tot)
    scattering_c0 = np.clip(
        c0 + object_map_mean + object_map_std * scattering_map,
        lower_amplitude,
        upper_amplitude,
    )
    scattering_rho0 = scattering_c0 / density_constant_multiplier
    # Define a sphere for a highly scattering region
    # object_map_center_x =26
    scattering_region = get_3d_circle(
        kgrid=kgrid,
        grid_size_points=grid_size_points,
        radius=object_map_radius,
        x_pos=object_map_center_x,
        y_nod=object_map_center_y,
        z_nod=object_map_center_z,
        number_scan_lines=number_scan_lines,
        tranducer_width=element_width,
        params=params,
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


def get_3d_circle(
    kgrid,
    grid_size_points,
    radius=8e-3,
    x_pos=32e-3,
    y_nod=2,
    z_nod=2,
    number_scan_lines=None,
    tranducer_width=None,
    params=None,
):
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
    # y_pos = kgrid.dy * kgrid.Ny / y_nod
    # y_pos = kgrid.dy * (kgrid.Ny + number_scan_lines * tranducer_width) / y_nod
    # z_pos = kgrid.dz * kgrid.Nz / z_nod
    # ball_center = np.round(Vector([x_pos, y_pos, z_pos]) / kgrid.dx)
    # x_pos = 32e-3
    # y_pos = 50 + (kgrid.dy * (kgrid.Ny) / 2)
    # z_pos = kgrid.dz * kgrid.Nz / 2
    # np.round(Vector([x_pos, y_pos, z_pos]) / kgrid.dx)
    # ball_center = Vector([172,80,80])
    radius = params['object_map_radius']
    x_pos = params['object_map_center_x']
    y_pos = params['object_map_center_y']
    z_pos = kgrid.dx * kgrid.Nz / 2
    grid_size_points = Vector([kgrid.Nx, kgrid.Ny + number_scan_lines * tranducer_width, kgrid.Nz])
    ball_center = np.round(Vector([x_pos, y_pos, z_pos]) /kgrid.dx)
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
    channel = 0
    # Visualize the sound speed map
    axs[0, 0].imshow(sound_speed_map[:, :, channel], cmap="gray")
    # Visualize the sound speed map
    axs[0, 0].set_title("Sound Speed Map")

    # Visualize the density map
    axs[0, 1].imshow(density_map[:, :, channel], cmap="gray")
    axs[0, 1].set_title("Density Map")

    # Visualize the scattering map
    axs[0, 2].imshow(scattering_map[:, :, channel], cmap="gray")
    axs[0, 2].set_title("Scattering Map")

    # Visualize the scattering c0 map
    axs[1, 0].imshow(scattering_c0[:, :, channel], cmap="gray")
    axs[1, 0].set_title("Scattering c0")

    # Visualize the scattering rho0 map
    axs[1, 1].imshow(scattering_rho0[:, :, channel], cmap="gray")
    axs[1, 1].set_title("Scattering rho0")
    # Visualize the scattering region
    scattering_region_map = np.zeros_like(sound_speed_map)
    scattering_region_map[scattering_region] = 1
    axs[1, 2].imshow(scattering_region_map[:, :, channel], cmap="gray")
    axs[1, 2].set_title("Scattering Region")
    # Visualize the final sound speed map
    axs[2, 1].imshow(final_sound_speed_map[:, :, channel], cmap="gray")
    axs[2, 1].set_title("Final Sound Speed Map")
    # Visualize the final density map
    axs[2, 2].imshow(final_density_map[:, :, channel], cmap="gray")
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
    args,
    transducer_width = 2
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
    # create the computational grid
    def make_grid(grid_size_points, grid_spacing_meters, c0, t_end):
        kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
        kgrid.makeTime(c0, t_end=t_end)
        return kgrid
    
    # Set the desired size of the image
    grid_size_meters = args["grid_size_meters"]
    number_scan_lines = args["number_scan_lines"]
    grid_size_points = Vector([kgrid.Nx, (kgrid.Ny + number_scan_lines * transducer_width), kgrid.Nz])
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])
    kgrid = make_grid(grid_size_points, grid_spacing_meters, c0, t_end=t_end)
    image_size = kgrid.size
    show_fig = args["show_fig"]
    # print("image_size", image_size)
    # # Create the axis variables
    x_axis = [0, image_size[0] * 1e3 * 1.1]  # [mm]
    y_axis = [-0.5 * image_size[1] * 1e3, 0.5 * image_size[1] * 1e3]  # [mm]
    # Define the horizontal axis
    # image_size = kgrid.size
    transducer_element_width = 1
    dy = grid_spacing_meters.y
    scale_factor = 2
    horz_axis = (np.arange(scan_lines_fund.shape[0]) * transducer_element_width * dy / scale_factor * 1e3)

    # make plotting non-blocking
    # plt.ion()
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
    
    # plt.imshow(
    #     scan_lines_fund.T,
    #     cmap="grey",
    #     aspect="auto",
    #     extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
    # )
    plt.imshow(
    scan_lines_fund.T,
    cmap="gray",
    aspect="auto",
    extent=[horz_axis[0], horz_axis[-1], 40, 5],  # Adjust the extent to match the MATLAB code
    # extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
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
    plt.savefig(
        f"{args['RESULTS_DIR']}/{args['experiment_name']}/visualization_maps.png"
    )
    if show_fig:
        plt.show()

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
    plt.savefig(
        f"{args['RESULTS_DIR']}/{args['experiment_name']}/rf_processing_result.png"
    )
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_data_using_phaser_template(
    scan_lines,
    scan_lines_fund,
    scan_lines_harm,
    steering_angles,
    c0,
    kgrid,
    medium,

):
    # Visualization
    image_size = [kgrid.Nx * kgrid.dx, kgrid.Ny * kgrid.dy]
    image_res = [256, 256]
    # b_mode_fund = scan_conversion(scan_lines_fund, steering_angles, image_size, c0, kgrid.dt, image_res)
    # b_mode_harm = scan_conversion(scan_lines_harm, steering_angles, image_size, c0, kgrid.dt, image_res)
    b_mode_fund = scan_lines_fund
    b_mode_harm = scan_lines_harm
    # Create the axis variables
    x_axis = [0, image_size[0] * 1e3]  # [mm]
    y_axis = [0, image_size[1] * 1e3]  # [mm]
    steering_angles = np.linspace(-30, 30, scan_lines.shape[0])


    # plt.ion()
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(
        scan_lines.T, aspect="auto", extent=[steering_angles[-1], steering_angles[0], y_axis[1], y_axis[0]], interpolation="none", cmap="gray"
    )
    plt.xlabel("Steering angle [deg]")
    plt.ylabel("Depth [mm]")
    plt.title("Raw Scan-Line Data")


    plt.subplot(132)
    plt.imshow(
        scan_lines_fund.T,
        aspect="auto",
        extent=[steering_angles[-1], steering_angles[0], y_axis[1], y_axis[0]],
        interpolation="none",
        cmap="bone",
    )
    plt.xlabel("Steering angle [deg]")
    plt.ylabel("Depth [mm]")
    plt.title("Processed Scan-Line Data")

    plt.subplot(133)
    plt.imshow(b_mode_fund, cmap="bone", aspect="auto", extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]], interpolation="none")
    plt.xlabel("Horizontal Position [mm]")
    plt.ylabel("Depth [mm]")
    plt.title("B-Mode Image")


    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(medium.sound_speed[..., kgrid.Nz // 2], aspect="auto", extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]])
    plt.xlabel("Horizontal Position [mm]")
    plt.ylabel("Depth [mm]")
    plt.title("Scattering Phantom")

    plt.subplot(132)
    plt.imshow(b_mode_fund, cmap="gray", aspect="auto", extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]], interpolation="none")
    plt.xlabel("Horizontal Position [mm]")
    plt.ylabel("Depth [mm]")
    plt.title("B-Mode Image")

    plt.subplot(133)
    plt.imshow(b_mode_harm, cmap="gray", aspect="auto", extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]], interpolation="none")
    plt.xlabel("Horizontal Position [mm]")
    plt.ylabel("Depth [mm]")
    plt.title("Harmonic Image")

    plt.show()
