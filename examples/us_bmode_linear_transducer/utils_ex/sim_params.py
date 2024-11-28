import os
import json
from kwave.data import Vector  # type: ignore


class Params:
    def __init__(self):
        # Initialization
        self.DATA_CAST = "single"
        self.RUN_SIMULATION = False
        self.experiment_name = "exp_9_circle_normal_noise"
        self.RESULTS_DIR = os.path.join(
            "C:/Users/CMME3/Documents/GitHub/k-wave-python/examples/us_bmode_linear_transducer",
            "results",
        )
        self.create_results_dir()
        self.debug = True  # True if you want to print debug messages
        self.show_fig = True  # True if you want to show the figures
        self.sc = 1
        # Grid Properties
        self.c0 = 1540
        self.rho0 = 1000
        self.pml_size_points = Vector([20, 10, 10])
        self.grid_size_points = Vector([int(256/self.sc), int(128/self.sc), int(128/self.sc)]) - 2 * self.pml_size_points
        self.grid_size_meters = 40e-3
        self.grid_spacing_meters = self.grid_size_meters / Vector(
            [self.grid_size_points.x, self.grid_size_points.x, self.grid_size_points.x]
        )
        # Medium Properties
        self.SOME_TIME_CONSTANT = 2.2
        self.alpha_coeff = 0.75
        self.alpha_power = 1.5
        self.BonA = 6
        # Transducer Settings
        self.source_strength = 1e6
        self.tone_burst_freq = 1.5e6 / self.sc
        self.tone_burst_cycles = 4
        self.number_scan_lines = int(96/self.sc)
        self.alpha_coeff = 0.75
        self.alpha_power = 1.5
        self.transducer_element_width = 1
        # Phantom Properties
        self.background_map_mean = 1
        self.background_map_std = 0.008
        self.object_map_mean = 25
        self.object_map_std = 75
        self.object_map_radius = 13e-3
        self.object_map_center_x = 20e-3
        self.object_map_center_y = 55e-3
        self.object_map_center_z = 25e-3
        self.density_constant_multiplier = 1.5
        self.lower_amplitude = 1400
        self.upper_amplitude = 1600
        # changed the ball center and grid properties inside the makeball func
        self.x_pos = 25e-3
        self.y_pos = 25e-3
        self.Nx_new = "kgrid.Nx"
        self.Ny_new = "kgrid.Ny + number_scan_lines * transducer.element_width /y_nod"
        self.Nz_new = "kgrid.Nz / 2"
        # Receiver Properties
        self.SHIFTING_SAMP_FREQ = 1
        self.BW_FUND = 100
        self.BW_HARM = 30
        self.COMPRESSION_RATIO = 3e0

    def create_results_dir(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, self.experiment_name), exist_ok=True)
