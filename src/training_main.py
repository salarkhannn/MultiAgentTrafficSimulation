from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

# Import custom modules for simulation, traffic generation, model, visualization, etc.
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

if __name__ == "__main__":

    # Import training configuration from a configuration file
    config = import_train_configuration(config_file='training_settings.ini')

    # Set up the SUMO traffic simulation command using the configuration details
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    # Create a unique directory path for saving the model and related files
    path = set_train_path(config['models_path_name'])

    # Initialize the neural network model using the TrainModel class
    Model = TrainModel(
        config['num_layers'],              # Number of hidden layers in the neural network
        config['width_layers'],            # Number of neurons per hidden layer
        config['batch_size'],              # Batch size for training
        config['learning_rate'],           # Learning rate for gradient descent
        input_dim=config['num_states'],    # Number of input features (state space dimension)
        output_dim=config['num_actions']   # Number of output features (action space dimension)
    )

    # Initialize the replay memory for experience storage and sampling
    Memory = Memory(
        config['memory_size_max'],  # Maximum size of the replay memory
        config['memory_size_min']   # Minimum size of memory to start training
    )

    # Initialize the traffic generator to simulate traffic conditions
    TrafficGen = TrafficGenerator(
        config['max_steps'],           # Maximum number of steps per simulation episode
        config['n_cars_generated']     # Number of cars generated in each episode
    )

    # Initialize the visualization tool for plotting results
    Visualization = Visualization(
        path,         # Directory path to save the plots
        dpi=96        # Dots per inch for image quality
    )
        
    # Set up the simulation environment for training
    Simulation = Simulation(
        Model,                       # Neural network model for Q-value approximation
        Memory,                      # Replay memory for experience replay
        TrafficGen,                  # Traffic generator for the simulation
        sumo_cmd,                    # SUMO command for traffic simulation
        config['gamma'],             # Discount factor for future rewards
        config['max_steps'],         # Maximum number of steps per episode
        config['green_duration'],    # Duration of green light in traffic signals
        config['yellow_duration'],   # Duration of yellow light in traffic signals
        config['num_states'],        # State space dimension
        config['num_actions'],       # Action space dimension
        config['training_epochs']    # Number of epochs for training per batch
    )

    # Initialize variables for the training loop
    episode = 0
    timestamp_start = datetime.datetime.now()  # Record the start time for the session
    
    # Training loop that runs for the total number of episodes
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        
        # Epsilon-greedy policy for balancing exploration and exploitation
        epsilon = 1.0 - (episode / config['total_episodes'])

        # Run the simulation for the current episode and train the model
        simulation_time, training_time = Simulation.run(episode, epsilon)
        
        # Display episode time statistics
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time + training_time, 1), 's')

        # Increment the episode counter
        episode += 1

    # Record and display the start and end time of the training session
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    # Save the trained model and its structure
    Model.save_model(path)

    # Copy the training configuration file to the session directory
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    # Save and plot cumulative reward data over episodes
    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')

    # Save and plot cumulative delay data over episodes
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')

    # Save and plot average queue length data over episodes
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
