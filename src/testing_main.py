from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

if __name__ == "__main__":
    # Load configuration settings
    test_config = import_test_configuration(config_file='testing_settings.ini')
    sumo_command = set_sumo(test_config['gui'], test_config['sumocfg_file_name'], test_config['max_steps'])
    model_directory, graph_output_path = set_test_path(test_config['models_path_name'], test_config['model_to_test'])

    # Initialize the traffic light model
    test_model = TestModel(
        input_dim=test_config['num_states'],
        model_path=model_directory
    )

    # Initialize the traffic route generator
    route_generator = TrafficGenerator(
        test_config['max_steps'], 
        test_config['n_cars_generated']
    )

    # Initialize the visualization tools
    test_visualization = Visualization(
        graph_output_path, 
        dpi=96
    )
        
    # Initialize the simulation environment
    traffic_simulation = Simulation(
        test_model,
        route_generator,
        sumo_command,
        test_config['max_steps'],
        test_config['green_duration'],
        test_config['yellow_duration'],
        test_config['num_states'],
        test_config['num_actions']
    )

    # Run the test simulation
    print('\n----- Starting Test Episode -----')
    runtime = traffic_simulation.run_testing_simulation(test_config['episode_seed'])
    print('Total Simulation Time:', runtime, 'seconds')

    # Save test information
    print("----- Test results saved at:", graph_output_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(graph_output_path, 'testing_settings.ini'))

    # Generate and save plots
    test_visualization.save_data_and_plot(
        data=traffic_simulation.reward_episode,
        filename='reward',
        xlabel='Action Step',
        ylabel='Reward'
    )
    test_visualization.save_data_and_plot(
        data=traffic_simulation.queue_length_episode,
        filename='queue',
        xlabel='Simulation Step',
        ylabel='Queue Length (vehicles)'
    )
