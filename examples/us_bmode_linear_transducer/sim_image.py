import json
from utils_ex.sim_params import Params
from utils_ex.helper import (
    get_transducer,
    make_grid,
    get_input_signal,
    solve_kspace_problem,
    process_simulation_data
)
from utils_ex.artifacts import get_phantom_data_circle

from kwave.kmedium import kWaveMedium  # type: ignore
from kwave.data import Vector  # type: ignore

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Vector):
            return obj.__dict__  # or any other way to serialize Vector
        return super().default(obj)

def save_json(data, filename = "parameters"):
    # Assuming args is an object with attributes
    with open(f'{data.RESULTS_DIR}/{data.experiment_name}/{filename}.json', 'w') as f:
        json.dump(data.__dict__, f, cls=CustomEncoder)
        
def run_simulation(args: Params):
    print("Running Simulation")
    # define the grid
    grid_size_points = args.grid_size_points
    grid_spacing_meters = args.grid_spacing_meters
    c0 = args.c0
    rho0 = args.rho0
    tone_burst_freq = args.tone_burst_freq
    tone_burst_cycles = args.tone_burst_cycles
    SOME_TIME_CONSTANT = args.SOME_TIME_CONSTANT
    source_strength = args.source_strength
    # create the computational grid
    print("Definign the grid")
    kgrid = make_grid(grid_size_points, grid_spacing_meters, c0, SOME_TIME_CONSTANT)
    # create the time array and input signal
    print("Defining the input signal")
    input_signal = get_input_signal(kgrid, tone_burst_freq, tone_burst_cycles, source_strength, c0, rho0)
    # define the transducer properties
    print("Defining the transducer")
    not_transducer,transducer = get_transducer(grid_size_points, input_signal, kgrid, c0)
    # define the medium properties
    medium = kWaveMedium(
        sound_speed=None,  # will be set later
        alpha_coeff=args.alpha_coeff,
        alpha_power=args.alpha_power,
        BonA=args.BonA,
    )
    # get the phantom
    print("Getting the phantom")
    phantom = get_phantom_data_circle(
                                    kgrid=kgrid,
                                    params = args.__dict__,
                                    debug = True
                                    )
    # get the phantom data and set the medium properties
    medium.sound_speed = phantom["sound_speed_map"]
    medium.density_map = phantom["density_map"]
    # run the simulation to get scan lines
    print("Running the simulation")
    scan_lines = solve_kspace_problem(
        kgrid=kgrid,
        medium=medium,
        args=args,
        sound_speed_map=phantom["sound_speed_map"],
        density_map=phantom["density_map"],
        grid_size_points=grid_size_points,
        transducer=transducer,
        not_transducer=not_transducer,        
    )
    # Process the scan lines using RF Signal Processing
    print("Processing the scan lines")
    processed_central_scan_line = process_simulation_data(kgrid=kgrid,
                            medium=medium,
                            input_signal=input_signal,
                            scan_lines=scan_lines,
                            tone_burst_freq=tone_burst_freq,
                            c0=c0,
                            args=args)
    # Save the parameters
    print("Saving the parameters")
    save_json(args)
    return processed_central_scan_line

if __name__ == "__main__":
    print("Hello World")
    args = Params()  # loading the parameters
    run_simulation(args = args)