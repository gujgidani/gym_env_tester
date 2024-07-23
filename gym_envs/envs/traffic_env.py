"""
This module contains the traffic light environment for the traffic light control problem.
"""
import os
import sys
import time

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from sumolib import checkBinary

import gym_envs.envs.traffic_light_support_functions as tlsf


class TrafficEnv(gym.Env):
    """
    The TrafficEnv class is a custom environment for the traffic light control problem.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, render_mode, starting_phases, phase_lengths, traffic_scale=1,
                 trafficlight_order=(5, 4, 0, 1, 3, 2), phase_number=6,
                 cycle_time=60, simulation_time=3600, phase_change_step=1):
        """
        The constructor for the TrafficEnv class.

        :parameter render_mode (str): The render mode for the environment. Must be either 'human' or
        'console'.
        :parameter starting_phases (np.array): The starting phases for the traffic lights.
        :parameter phase_lengths (np.array): The lengths of each phase for the traffic lights.
        :parameter traffic_scale (int): The scale of the traffic in the simulation.
        :parameter trafficlight_order (tuple of int): The order of the traffic lights in the
        simulation.
        :parameter phase_number (int): The number of phases in the simulation.
        :parameter cycle_time (int): The cycle time for the traffic lights.
        :parameter simulation_time (int): The simulation time for the traffic lights.
        :parameter phase_change_step (int): The step for changing the phase.
        :return None
        """
        self.traffic_scale = traffic_scale
        self.traci_order = trafficlight_order
        self.cycle_time = cycle_time
        self.phase_plan = np.empty((5, phase_number, cycle_time), dtype=np.int8)
        self.initial_phase_plan = tlsf.generate_phase_plan(starting_phases, phase_lengths)
        self.time_limit = simulation_time
        self.current_step = 0
        self.edge_list = ("660942467#1","-24203041#0","660942464")
        # Create a np array with 5 initial Phaseplans
        for i in range(5):
            self.phase_plan[i] = self.initial_phase_plan

        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

        if render_mode == 'console':
            #self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo"
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo"
            else:
                self.sumo_binary = '/usr/bin/sumo'
        else:
            #self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo-gui"
            #self.sumo_binary = checkBinary('sumo-gui')
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo-gui"
            else:
                self.sumo_binary = '/usr/bin/sumo-gui'
        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            "sumo_files/osm.sumocfg",
            "--start",
            "-e", str(simulation_time),
            "--quit-on-end",
            "--scale", str(traffic_scale)
        ]
        traci.start(self.sumo_cmd)

        self.observation_space = spaces.Dict({
            # Megkérdezni Tamásékat a max értékről
            "occupancy": spaces.Box(low=0.0, high=100.0, shape=(6,), dtype=np.float32),
            "vehicle_count": spaces.Box(low=0, high=7200, shape=(6,), dtype=np.int32),
            "last_five_phaseplan": spaces.MultiDiscrete(
                nvec=np.full((5,phase_number, cycle_time), 4), dtype=np.int8,
                start=np.full((5,phase_number, cycle_time), 1))
        })

        self.actions = tlsf.generate_phase_combinations(5, 36,step = phase_change_step)
        self.action_space = spaces.Discrete(len(self.actions))

    def _get_obs(self):
        """
        Returns the observation for the environment.
        :return: The observation for the environment. Occupancy, vehicle count and the last
        five phase plans.
        """
        detectors = traci.inductionloop.getIDList()
        occopancy = []
        vehicle_count = []
        for detector in detectors:
            occopancy.append(traci.inductionloop.getLastIntervalOccupancy(detector))
            vehicle_count.append(traci.inductionloop.getLastIntervalVehicleNumber(detector))
        return {
            "occupancy": np.array(occopancy, dtype=np.float32),
            "vehicle_count": np.array(vehicle_count, dtype=np.int32),
            "last_five_phaseplan": self.phase_plan
        }

    def _get_info(self):
        """
        Returns the information for the environment.
        :return: The information for the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        :param seed: The seed for the environment.
        :param options: The options for the environment.
        :return: The observation for the environment.
        """
        #super().reset(seed=seed)
        #try:
        #    traci.close()
        #    time.sleep(0.001)
        #except ConnectionError:
        #    pass
        self.current_step = 0

        self.sumo_cmd[-1] = str(self.np_random.integers(2, 6) / 4.0)
        #traci.start(self.sumo_cmd, port=8813)
        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)
        for i in range(0, 300):
            traci.simulationStep()
            self.current_step += 1
        for i in range(5):
            self.phase_plan[i] = self.initial_phase_plan

        self.edge_list = traci.edge.getIDList()

        return self._get_obs(), {}

    def step(self, action):
        """
        Takes a step in the environment.
        :param action: The action to take.
        :return: The observation, reward, done and info for the environment.
        """
        #Rotate the phase plan and put the new phase to the end
        self.phase_plan = np.roll(self.phase_plan, 1, axis=1)
        #self.phase_plan[:, -1] = tlsf.change_phase_plan(self.actions[action])
        self.phase_plan[-1] = tlsf.change_phase_plan(self.actions[action[0]], self.cycle_time)
        travel_time = 0
        mean_speed = 0
        #TODO imeplementálni a közbeeső idő checket

        for j in range (0,5):
            for i in range(0, self.cycle_time):
                traci.simulationStep()
                self.current_step += 1
                tls = traci.trafficlight.getIDList()
                traci.trafficlight.setRedYellowGreenState(
                    tls[0], tlsf.get_phase_column_for_step(self.phase_plan[-1], i, self.traci_order)
                )
                for edge in self.edge_list:
                    mean_speed += traci.edge.getLastStepMeanSpeed(edge)/ 13.89 / len(self.edge_list)
        obs = self._get_obs()
        # The reward is the negative of the traveltimes
        # Reward = átlag sebessség edge-kre
        reward = mean_speed / self.cycle_time / 5

        #done = traci.simulation.getTime() / 1000 > self.time_limit
        done = self.current_step > self.time_limit
        #print("Current time: ", self.current_step, " seconds.")
        info = {}
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()
