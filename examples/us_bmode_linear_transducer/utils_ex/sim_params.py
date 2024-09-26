import os
import json
from kwave.data import Vector  # type: ignore


class Params:
    def __init__(self):
        # Initialization
        self.DATA_CAST = "single"
        self.RUN_SIMULATION = True
        self.experiment_name = "exp_1_circle_normal_noise"
        self.RESULTS_DIR = os.path.join(
            r"C:\Users\CMME3\Documents\GitHub\k-wave-python\examples\us_bmode_linear_transducer",
            "results",
        )

        self.create_results_dir()
        self.debug = True
        self.show_fig = False
        # Grid Properties
        self.c0 = 1540
        self.rho0 = 1000
        self.pml_size_points = Vector([20, 10, 10])
        self.grid_size_points = Vector([256, 128, 128]) - 2 * self.pml_size_points
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
        self.tone_burst_freq = 1.5e6
        self.tone_burst_cycles = 4
        self.number_scan_lines = 96
        self.alpha_coeff = 0.75
        self.alpha_power = 1.5
        # Phantom Properties
        self.background_map_mean = 1
        self.background_map_std = 0.008
        self.object_map_mean = 25
        self.object_map_std = 75
        self.object_map_radius = 8e-3
        self.object_map_center_x = 32e-3
        self.object_map_center_y = 2
        self.object_map_center_z = 2
        self.density_constant_multiplier = 1.5
        self.lower_amplitude = 1400
        self.upper_amplitude = 1600
        # Receiver Properties
        self.SHIFTING_SAMP_FREQ = 1
        self.BW_FUND = 100
        self.BW_HARM = 30
        self.COMPRESSION_RATIO = 15

    def create_results_dir(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, self.experiment_name), exist_ok=True)

