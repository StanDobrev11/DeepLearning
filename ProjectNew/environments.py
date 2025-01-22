from typing import Tuple, Dict, Optional, Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten

import numpy as np

from ProjectNew.vessels import OwnShip, Target, StaticObject


class MarineEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
            self,
            initial_lat: float = 0.0,
            initial_lon: float = 0.0,
            env_range: int = 30,
            timescale: int = 1,
            continuous: bool = False,
            max_turn_angle: int = 10,
            max_speed_change: float = 0.1,
            waypoint_reach_threshold: float = 0.2,
            render_mode=None,
            training_stage: int = 1,  # defines different training stages, 0 for no training
            total_targets: Optional[int] = None
    ):
        super(MarineEnv, self).__init__()
        self.training_stage = training_stage
        self.total_targets = total_targets

        # initialize the environment bounds
        self.lat_bounds: Tuple[float, float] = (initial_lat, initial_lat + env_range / 60)
        self.lon_bounds: Tuple[float, float] = (initial_lon, initial_lon + env_range / 60)

        # define the timescale and scaling of real word time to simulation time
        self.timescale = timescale

        # initialize action space
        if continuous:
            # the agent can continuously adjust course and speed
            self.max_turn_angle = max_turn_angle  # scaling the turn
            self.max_speed_change = max_speed_change  # scaling the speed change
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        # define the observation space
        self.observation_space = spaces.Dict({
            'own_ship': spaces.Dict({
                'course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                'speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                'wp_distance': spaces.Box(low=0, high=env_range, shape=(), dtype=np.float32),
                'wp_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
            }),
            'targets': spaces.Tuple([
                spaces.Dict({
                    'bcr': spaces.Box(low=-50, high=50, shape=(), dtype=np.float32),
                    'cpa': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_distance': spaces.Box(low=0, high=env_range, shape=(), dtype=np.float32),
                    'target_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
                    'target_relative_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_relative_speed': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                    'tbcr': spaces.Box(low=-10, high=100, shape=(), dtype=np.float32),
                    'tcpa': spaces.Box(low=-60, high=100, shape=(), dtype=np.float32),
                }) for _ in range(3)  # assuming 3 targets are considered dangerous
            ])
        })

        # initialize the state
        self.observation = np.zeros(flatten_space(self.observation_space).shape)

        # initialize own ship
        self.own_ship = OwnShip()

        # initialize the wp and wp reach threshold
        self.waypoint = StaticObject()  # 2 elements, latitude and longitude
        self.waypoint_reach_threshold = waypoint_reach_threshold  # in nautical miles, terminates an episode

        # pygame setup
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode: {render_mode}. Available modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

        self.window_size = 600  # Pixels for visualization
        self.scale = self.window_size / ((self.lat_bounds[1] - self.lat_bounds[0]) * 60)  # Pixels per NM
        self.window = None
        self.clock = None
        self.vessel_size = 5  # Vessel radius in pixels

    def step(self, action):

        # extract own ship params from the current state
        course, speed, last_wp_distance, last_wp_relative_bearing = self.observation[:4]

        # update the params based on action
        if 'Discrete' in str(type(env.action_space)):
            course_change = 0
            speed_change = 0
            if action == 0:  # do nothing, keep course and speed
                pass
            elif action == 1:  # turn port (left) by 1 degree
                course_change = - 1
            elif action == 2:  # turn starboard (right) by 1 degree
                course_change = 1
            elif action == 3:  # reduce speed
                speed_change = -0.1
            elif action == 4:  # increase speed
                speed_change = 0.1
        else:
            course_change = np.clip(action[0] * self.max_turn_angle, -self.max_turn_angle, self.max_turn_angle)
            speed_change = action[1] * self.max_speed_change

        # update and apply own ship parameters
        course = (course + course_change) % 360
        speed += speed_change
        self.own_ship.course = course
        self.own_ship.speed = speed

        # update next position
        self.own_ship.update_position(time_interval=self.timescale, clip_lat=self.lat_bounds, clip_lon=self.lon_bounds)

        # calculate current distance and current relative bearing to wp
        current_wp_distance = self.own_ship.calculate_distance(self.waypoint)
        current_wp_rel_bearing = self.own_ship.calculate_relative_bearing(self.waypoint)

        # generate own state data
        own_ship_data = {
            "course": course,
            "speed": speed,
            "wp_distance": current_wp_distance,
            "wp_relative_bearing": current_wp_rel_bearing,
        }

        if self.own_ship.detected_targets:
            # Move/update all detected targets
            for target in self.own_ship.detected_targets:
                target.update_position(time_interval=self.timescale, clip_lat=self.lat_bounds, clip_lon=self.lon_bounds)
                self.own_ship.update_target(target)

            # Sort detected targets by dangerous coefficient (CPA ** 2 * TCPA) and take the top 3
            self.own_ship.dangerous_targets = sorted(
                self.own_ship.detected_targets, key=lambda x: x.cpa ** 2 * x.tcpa
            )[:3]

            # initialize empty targets list with zero-filled entries
            targets_data = [
                {
                    'bcr': 0.0,
                    'cpa': 0.0,
                    'target_course': 0.0,
                    'target_distance': 0.0,
                    'target_relative_bearing': 0.0,
                    'target_relative_course': 0.0,
                    'target_relative_speed': 0.0,
                    'target_speed': 0.0,
                    'tbcr': 0.0,
                    'tcpa': 0.0,
                } for _ in range(3)
            ]

            # fill in detected targets (up to 3)
            for i, target in enumerate(self.own_ship.dangerous_targets):
                target.is_dangerous = True
                targets_data[i] = {
                    'bcr': target.bcr,
                    'cpa': target.cpa,
                    'target_course': target.course,
                    'target_distance': target.distance,
                    'target_relative_bearing': target.relative_bearing,
                    'target_relative_course': target.relative_course,
                    'target_relative_speed': target.relative_speed,
                    'target_speed': target.speed,
                    'tbcr': target.tbcr,
                    'tcpa': target.tcpa,
                }

        else:
            # no detected targets, return all-zero target data
            targets_data = [
                {
                    'bcr': 0.0,
                    'cpa': 0.0,
                    'target_course': 0.0,
                    'target_distance': 0.0,
                    'target_relative_bearing': 0.0,
                    'target_relative_course': 0.0,
                    'target_relative_speed': 0.0,
                    'target_speed': 0.0,
                    'tbcr': 0.0,
                    'tcpa': 0.0,
                } for _ in range(3)
            ]

        # construct the final observation
        raw_observation = {
            "own_ship": own_ship_data,
            "targets": targets_data
        }

        self.observation = flatten(self.observation_space, raw_observation)

        return self.observation

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        # seed for random number generator
        super().reset(seed=seed)

        own_ship_data = {}
        targets_data = {}

        # TODO code the state for testing, not training, provided the target count
        if self.training_stage == 0:
            pass

        elif self.training_stage == 1:  # training for wp tracking, no targets

            # placing the vessel in the center of the env
            self.own_ship.lat, self.own_ship.lon = np.mean(self.lat_bounds), np.mean(self.lon_bounds)

            # random initial speed, minimum 7 kn
            self.own_ship.speed = np.random.uniform(low=7, high=self.own_ship.max_speed)

            # random initial course
            self.own_ship.course = np.random.uniform(low=0, high=360)

            # place the waypoint
            while True:
                waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
                waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
                distance_to_waypoint = self.own_ship.calculate_distance((waypoint_lat, waypoint_lon))
                # Ensure waypoint is 3 NM to 20 NM away from the vessel
                if 3.0 < distance_to_waypoint < 20:
                    break

            self.waypoint.lat, self.waypoint.lon = waypoint_lat, waypoint_lon

            own_ship_data = {
                "course": self.own_ship.course,
                "speed": self.own_ship.speed,
                "wp_distance": self.own_ship.calculate_distance(self.waypoint),
                "wp_relative_bearing": self.own_ship.calculate_relative_bearing(self.waypoint),
            }

            targets_data = [{
                'bcr': 0.0,
                'cpa': 0.0,
                'target_course': 0.0,
                'target_distance': 0.0,
                'target_relative_bearing': 0.0,
                'target_relative_course': 0.0,
                'target_relative_speed': 0.0,
                'target_speed': 0.0,
                'tbcr': 0.0,
                'tcpa': 0.0,
            } for _ in range(3)]

        # TODO set the state for training with targets
        else:
            pass

        raw_observation = {
            "own_ship": own_ship_data,
            "targets": targets_data
        }

        self.observation = flatten(self.observation_space, raw_observation)

        return self.observation, {}


if __name__ == '__main__':
    env = MarineEnv(continuous=True)
    # print(env.observation_space)
    # print(flatten_space(env.observation_space))
    # print(env.observation)
    # print(env.own_ship)
    env.reset()
    # print(type(env.action_space))
    # print(env.step((0.7, 0.5)))
    # print(env.observation)
    target_ship = Target(position=(0.05, 0.05), course=270, speed=5, min_speed=5, max_speed=20)
    env.observation[0] = 0
    env.observation[1] = 10
    env.own_ship.lat = 0.0
    env.own_ship.lon = 0.0
    env.own_ship.course = 0.0
    env.own_ship.speed = 10.0
    env.own_ship.min_speed = 5
    env.own_ship.max_speed = 20
    env.own_ship.detected_targets = [target_ship]
    print(env.step((0, 0)))
    print(env.step((0, 0)))
    print(env.step((0, 0)))
