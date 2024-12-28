import traci
import numpy as np
import random
import timeit
import os

# Traffic light phase constants based on environment.net.xml
PHASE_NS_GREEN = 0  # North-South green light (Action 0: 00)
PHASE_NS_YELLOW = 1  # North-South yellow light
PHASE_NSL_GREEN = 2  # North-South left green light (Action 1: 01)
PHASE_NSL_YELLOW = 3  # North-South left yellow light
PHASE_EW_GREEN = 4  # East-West green light (Action 2: 10)
PHASE_EW_YELLOW = 5  # East-West yellow light
PHASE_EWL_GREEN = 6  # East-West left green light (Action 3: 11)
PHASE_EWL_YELLOW = 7  # East-West left yellow light


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._step = 0
        self._reward_episode = []
        self._queue_length_episode = []
        self._waiting_times = {}

    def run_testing_simulation(self, episode):
        """
        runs the simulation for a specific episode.
        """
        start_time = timeit.default_timer()

        # Generate the traffic routes and initialize SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulation in progress...")

        # Initialize variables for the simulation loop
        self._step = 0
        previous_total_wait = 0
        previous_action = -1  # Placeholder for the initial action

        while self._step < self._max_steps:
            # Get the current state representation of the intersection
            current_state = self._get_state()

            # Calculate the reward by measuring the change in total waiting time
            total_waiting_time = self._get_waiting_times()
            reward = previous_total_wait - total_waiting_time

            # Determine the next traffic light phase based on the current state
            action = self._choose_action(current_state)

            # If the action changes, set a yellow transition phase
            if self._step > 0 and action != previous_action:
                self._set_yellow_phase(previous_action)
                self._simulate_test(self._yellow_duration)

            # Activate the selected green phase
            self._set_green_phase(action)
            self._simulate_test(self._green_duration)

            # Save values for tracking and accumulate reward
            previous_action = action
            previous_total_wait = total_waiting_time
            self._reward_episode.append(reward)

        traci.close()
        simulation_duration = round(timeit.default_timer() - start_time, 1)
        return simulation_duration

    def _simulate_test(self, steps_todo):
        """
        Advances the simulation in SUMO for the specified steps.
        """
        steps_remaining = min(steps_todo, self._max_steps - self._step)
        while steps_remaining > 0:
            traci.simulationStep()  # Perform a single simulation step in SUMO
            self._step += 1
            steps_remaining -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _get_waiting_times(self):
        """
        Computes the cumulative waiting time for vehicles approaching the intersection.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            if road_id in incoming_roads:
                self._waiting_times[vehicle_id] = wait_time
            elif vehicle_id in self._waiting_times:
                del self._waiting_times[vehicle_id]  # Vehicle has left monitored roads
        return sum(self._waiting_times.values())

    def _choose_action(self, state):
        """
        Selects the optimal action for the current state based on the trained model.
        """
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Configures the yellow phase based on the previous green phase.
        """
        yellow_phase_id = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_id)

    def _set_green_phase(self, action_number):
        """
        Configures the green light phase corresponding to the chosen action.
        """
        phase_mapping = {
            0: PHASE_NS_GREEN,
            1: PHASE_NSL_GREEN,
            2: PHASE_EW_GREEN,
            3: PHASE_EWL_GREEN,
        }
        traci.trafficlight.setPhase("TL", phase_mapping[action_number])

    def _get_queue_length(self):
        """
        Counts the number of halted vehicles at the intersection.
        """
        halted_north = traci.edge.getLastStepHaltingNumber("N2TL")
        halted_south = traci.edge.getLastStepHaltingNumber("S2TL")
        halted_east = traci.edge.getLastStepHaltingNumber("E2TL")
        halted_west = traci.edge.getLastStepHaltingNumber("W2TL")
        return halted_north + halted_south + halted_east + halted_west

    def _get_state(self):
        """
        Constructs the intersection state array using vehicle positions.
        """
        state = np.zeros(self._num_states)
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:
            lane_position = traci.vehicle.getLanePosition(vehicle_id)
            lane_position_inverted = 750 - lane_position  # Convert to distance from the light
            lane_id = traci.vehicle.getLaneID(vehicle_id)

            # Map lane position to grid cells
            if lane_position_inverted < 7:
                cell_index = 0
            elif lane_position_inverted < 14:
                cell_index = 1
            elif lane_position_inverted < 21:
                cell_index = 2
            elif lane_position_inverted < 28:
                cell_index = 3
            elif lane_position_inverted < 40:
                cell_index = 4
            elif lane_position_inverted < 60:
                cell_index = 5
            elif lane_position_inverted < 100:
                cell_index = 6
            elif lane_position_inverted < 160:
                cell_index = 7
            elif lane_position_inverted < 400:
                cell_index = 8
            else:
                cell_index = 9

            # Determine the lane group
            lane_map = {
                "W2TL_0": 0, "W2TL_1": 0, "W2TL_2": 0, "W2TL_3": 1,
                "N2TL_0": 2, "N2TL_1": 2, "N2TL_2": 2, "N2TL_3": 3,
                "E2TL_0": 4, "E2TL_1": 4, "E2TL_2": 4, "E2TL_3": 5,
                "S2TL_0": 6, "S2TL_1": 6, "S2TL_2": 6, "S2TL_3": 7,
            }
            lane_group = lane_map.get(lane_id, -1)

            if lane_group != -1:
                state_index = lane_group * 10 + cell_index
                state[state_index] = 1

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
